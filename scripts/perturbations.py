"""Script to run perturbation analysis"""

import json
import pandas as pd
import sklearn
from torch import nn
from tqdm import tqdm
import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
import random


path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
sys.path.append(path + '/continualdatasets')
sys.path.append(path + '/pytorch-hessian-eigenthings')
sys.path.append(path + '/backbone')
sys.path.append(path + '/models')


from continualdatasets import NAMES as DATASET_NAMES
from continualdatasets import ContinualDataset, get_dataset
from models import get_all_models, get_model
from utils.args import add_management_args
from utils.best_args import best_args
from utils.conf import set_random_seed, WANDBKEY
from utils.continual_training import train as ctrain
from utils.distributed import make_dp
from utils.training import train
from utils.main import parse_args
from utils.status import ProgressBar
from pytorch_hessian_eigenthings.hessian_eigenthings import compute_hessian_eigenthings

def get_Hessian_eigenthings(self, dataloader, n_h_samples, num_eigenthings):
        """Uses the pytorch Hessian eigenthings library to compute the first n eigenvectors and eigenvalues 
        of the model Hessian matrix."""
        #subsampling the data 
        if n_h_samples is not None:
            n_sample = min(n_h_samples, len(dataloader.dataset))
            self.log('Sample',n_h_samples,' batch for estimating the Hessian matrix.')
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample) # samples indices
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind) #creating a dataset with these samples
            dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, num_workers=2, batch_size=128)
        else: self.log('Using the full dataset for estimating the task Hessian matrix.')

        self.log(f"Computing Hessian first {num_eigenthings} eigenthings")

        eigenvals, eigenvecs = compute_hessian_eigenthings(
                        model=self,
                        dataloader=dataloader,
                        loss=self.criterion,
                        num_eigenthings=num_eigenthings,
                        full_dataset=True,
                        mode='lanczos', #alternatives: power_iter, lanczos
                        use_gpu=self.gpu,
                        #max_possible_gpu_samples=, #TODO: check?
                        fp16=False 
                    )

        return torch.Tensor(eigenvals.copy()), torch.Tensor(eigenvecs.copy().T)


def apply_perturbations(model, P, loss, acc, theta_star, radiuses= [0.0], average=False):
    """Applies perturbations (P matrix columns) to the model's parameters and records 
     forgetting on loss and accuracy. """
    E_tilde = []
    for r in radiuses: 
        print(f"------ Radius {r} ------")
        for i in range(P.size(1)):  
            if average: radius_list = []   
            _theta_tilde = theta_star + r*P[:,i].to(theta_star.device)
            vector_to_parameters(_theta_tilde, model.trainable_parameters())
            acc_tilde, loss_tilde = model.validate(dataloader)
            if not average: E_tilde.append([r,i,(loss_tilde - loss ),(acc - acc_tilde)])
            else: radius_list.append([(loss_tilde - loss ),(acc - acc_tilde)])
        if average: 
            mean_E_radius = np.abs(np.array(radius_list)).mean(axis=0)
            E_tilde.append([r,mean_E_radius[0],mean_E_radius[1]])
    return E_tilde
def do_perturbation_analysis(model, dataloader, radiuses, num_eigenthings=10):
    """Performs the perturbation analysis for the given model.
    Returns the forgetting table.
    Note: the model has to be loaded to the task checkpoint and the dataloader must belong to the same task.""" 
    # recording accuracy and loss before perturbations
    acc_star, loss_star = model.validate(dataloader)
    theta_star = parameters_to_vector(model.trainable_parameters()) #TODO
    # collect eigenvectors 
    # take the first K eigenvectors/values 
    L, Q = model.get_Hessian_eigenthings(dataloader, n_h_samples=2000, 
                                         num_eigenthings=num_eigenthings)
    pos_eig_idx = L > 0
    Q_p = Q[:,pos_eig_idx]
    E_delta = apply_perturbations(model, Q_p, 
                                  loss=loss_star, acc=acc_star, theta_star=theta_star, 
                                  radiuses= radiuses)
    # now random perturbations 
    # random perturbations 
    R_p = torch.randn_like(Q_p)
    for i in range(R_p.size(1)):
        R_p[:,i]/=(torch.norm(R_p[:,i])+10e-7)
    E_eta = apply_perturbations(model, R_p, 
                                loss=loss_star, acc=acc_star, theta_star=theta_star, 
                                radiuses= radiuses, average=True)
    
    E_delta_df = pd.DataFrame(E_delta, columns=['radius', 'eigen-index', 'E_loss', 'E_acc'])
    E_eta_df = pd.DataFrame(E_eta, columns=['radius', 'E_loss_rnd', 'E_acc_rnd'])
    merge_df = pd.merge(E_delta_df, E_eta_df, on="radius")[['radius', 'eigen-index','E_loss', 'E_acc', 'E_loss_rnd', 'E_acc_rnd']]
    merge_df['E_loss_ratio']= merge_df['E_loss']/(merge_df['E_loss_rnd']+10e-10)
    merge_df['E_acc_ratio']= merge_df['E_acc']/(merge_df['E_acc_rnd']+10e-10)
    res_df = merge_df[['radius','eigen-index','E_loss','E_acc','E_loss_ratio','E_acc_ratio']]
    return res_df



args = parse_args()

torch.set_default_dtype(torch.float32)

#instantiating model and dataset 
dataset = get_dataset(args)
args.batch_size = dataset.get_batch_size()
backbone = dataset.get_backbone()
loss = dataset.get_loss()
model = get_model(args, backbone, loss, dataset.get_transform())
model.net.to(model.device)
model.net.eval()

progress_bar = ProgressBar(verbose=not args.non_verbose)

for t in range(dataset.N_TASKS):
    train_loader, test_loader = dataset.get_data_loaders()
    model.net.eval()
    avg_loss = 0.0
    for i, data in enumerate(train_loader):
        if args.debug_mode and i > 3: # only 3 batches in debug mode
            break
        if hasattr(dataset.train_loader.dataset, 'logits'):
            inputs, labels, not_aug_inputs, logits = data
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)
            logits = logits.to(model.device)
            loss = model.meta_observe(inputs, labels, not_aug_inputs, logits)
        else:
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(model.device), labels.to(
                model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)
            loss = model.meta_observe(inputs, labels, not_aug_inputs)

        assert not math.isnan(loss)
        progress_bar.prog(i, len(train_loader), epoch, t, loss)
        avg_loss += loss
     

path = base_path_checkpoints() +  \
                dataset.SETTING + "/" + \
                    dataset.NAME + "/" + \
                        model.NAME
            model.load_checkpoint(state=state, path=path, task=t)


idxs = list(range(10, 15)) # different seeds 
print(f"Starting perturbation analysis of {len(models)} models")
perturbations_df = pd.DataFrame(columns=[['radius','eigen-index','E_loss','E_acc','E_loss_ratio','E_acc_ratio','seed','task','P','train_P','dataset','exp_type']])
radiuses = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

for idx in tqdm(idxs): 
    # for each model, we save the perturbation dataframe to file
    try: perturbations_df = pd.read_csv(os.path.join(common_output_dir, f"perturbations_df_{ID}.csv"), index_col=0)
    except FileNotFoundError: pass
    model = models[idx]
    T = len(model.config['task_names'])
    P = model.count_parameters(train=False)
    train_P = model.count_parameters(train=True)
    for t in range(T): # different tasks
        chkpt_name = f"/chkpt_task_{t}"
        model.load_checkpoint(path=model.config['exp_dir']+chkpt_name, task=t)


        dataloader = torch.utils.data.DataLoader(datasets[model.config['task_names'][t]], # make sure it's the right datasets
                                                    batch_size=100, shuffle=True, num_workers=2)
        
        res_df = do_perturbation_analysis(model, dataloader, radiuses, num_eigenthings=10)
        res_df['seed'] = model.config['seed']
        res_df['task'] = t
        res_df['P'] = P 
        res_df['train_P'] = train_P
        res_df['dataset'] = model.config['dataset']
        res_df['exp_type'] = model.config['exp_type']
        perturbations_df = pd.concat([res_df, perturbations_df])
    print(f"Saving for model {idx}")
    perturbations_df.to_csv(os.path.join(common_output_dir,f"perturbations_df_{ID}.csv"))