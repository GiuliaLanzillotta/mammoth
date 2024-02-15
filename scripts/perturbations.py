"""Script to run perturbation analysis"""

import math
import pandas as pd
from tqdm import tqdm
import numpy  # needed (don't change it)
import os
import sys
import torch
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
sys.path.append(path + '/continualdatasets')
sys.path.append(path + '/pytorch-hessian-eigenthings')
sys.path.append(path + '/pytorch-hessian-eigenthings/hessian_eigenthings')
sys.path.append(path + '/backbone')
sys.path.append(path + '/models')


from continualdatasets import NAMES as DATASET_NAMES
from continualdatasets import ContinualDataset, get_dataset
from models import get_all_models, get_model
from utils.args import add_management_args
from utils.best_args import best_args
from utils.conf import base_path, base_path_checkpoints, set_random_seed, WANDBKEY
from utils.continual_training import train as ctrain
from utils.distributed import make_dp
from utils.training import train
from utils.main import parse_args
from utils.status import ProgressBar
from pytorch_hessian_eigenthings.hessian_eigenthings import compute_hessian_eigenthings


RADIUS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000, 1000000]
SAMPLES = 2000

def get_Hessian_eigenthings(model, dataloader, n_h_samples, num_eigenthings):
        """Uses the pytorch Hessian eigenthings library to compute the first n eigenvectors and eigenvalues 
        of the model Hessian matrix."""
        #subsampling the data 
        if n_h_samples is not None:
            n_sample = min(n_h_samples, len(dataloader.dataset))
            print('Sample',n_h_samples,' batch for estimating the eigenvectors.')
            random_indices = numpy.random.choice(list(range(n_sample)), size=n_sample, replace=False) # samples indices
            subdata = torch.utils.data.Subset(dataloader.dataset, random_indices) #creating a dataset with these samples
            dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, num_workers=4, batch_size=128)
        else: print('Using the full dataset for estimating the eigenvectors.')

        print(f"Computing Hessian first {num_eigenthings} eigenthings")
        eigenvals, eigenvecs = compute_hessian_eigenthings(
                        model=model,
                        dataloader=dataloader,
                        loss=model.loss,
                        num_eigenthings=num_eigenthings,
                        full_dataset=True,
                        mode='lanczos', #alternatives: power_iter, lanczos
                        use_gpu=model.device!='gpu',
                        max_possible_gpu_samples=256, #might be too high for some GPUs
                        fp16=False 
                    )
        
        return torch.Tensor(eigenvals.copy()), torch.Tensor(eigenvecs.copy().T)


def apply_perturbations(model, dataloader, mat, loss, theta_star, radiuses=[0.0], test=False):
    """Applies perturbations (P matrix columns) to the model's parameters and records 
     forgetting on loss and accuracy. """
    E_tilde = []
    for r in radiuses: 
        print(f"------ Radius {r} ------")
        for i in range(mat.size(1)): # fo all the eigendirections  
            _theta_tilde = theta_star + r*mat[:,i].to(theta_star.device)
            vector_to_parameters(_theta_tilde, model.parameters()) #TODO: does this work?
            loss_tilde = compute_loss_on_dataset(model, dataloader, test=test)
            E_tilde.append([r,i,(loss_tilde - loss)])
    return E_tilde

def compute_loss_on_dataset(model, dataloader, test=False):
    """ Returns average loss on the dataset given. """
    progress_bar = ProgressBar(verbose=not args.non_verbose)

    avg_loss = 0.0
    for i, data in enumerate(dataloader):
        if hasattr(dataloader.dataset, 'logits'):
            inputs, labels, not_aug_inputs, logits = data
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            not_aug_inputs = not_aug_inputs.to(model.device)
    
        else:
            if not test: 
                inputs, labels, not_aug_inputs = data
                inputs = not_aug_inputs
            else: 
                inputs, labels = data

            inputs, labels = inputs.to(model.device), labels.to(model.device)

        with torch.no_grad():
            outputs = model.forward(inputs)
            loss = model.loss(outputs, labels).item()

        if math.isnan(loss): 
            print(f"Loss is {loss}")
            return numpy.nan
        progress_bar.prog(i, len(train_loader), 0, 0, loss)
        avg_loss += loss

    return avg_loss/len(train_loader)


def do_perturbation_analysis(model, dataloader, num_eigenthings=10, test=False):
    """Performs the perturbation analysis for the given model.
    Returns the forgetting table.
    Note: the model has to be loaded to the task checkpoint and the dataloader must belong to the same task.""" 
    # recording accuracy and loss before perturbations
    loss_star = compute_loss_on_dataset(model, dataloader, test=test)
    theta_star = parameters_to_vector(model.parameters()).detach()
    # collect eigenvectors 
    # take the first K eigenvectors/values 
    L, Q = get_Hessian_eigenthings(model, dataloader, n_h_samples=SAMPLES, num_eigenthings=num_eigenthings)

    Q_p = Q[:,:] # only select positive eigenvalues --- why? 
    E_delta = apply_perturbations(model, dataloader, Q_p, loss=loss_star, theta_star=theta_star, radiuses=RADIUS, test=test)
    # now random perturbations 
    # random perturbations 
    R_p = torch.randn_like(Q_p)
    R_p/=(torch.linalg.norm(R_p, ord=2, dim=0)+10e-7)
    E_eta = apply_perturbations(model, dataloader, R_p, loss=loss_star, theta_star=theta_star, radiuses= RADIUS, test=test)
    
    E_delta_df = pd.DataFrame(E_delta, columns=['radius', 'eigen-index', 'E_loss'])
    E_eta_df = pd.DataFrame(E_eta, columns=['radius', 'random-index', 'E_loss_rnd'])
    merge_df = E_delta_df.merge(E_eta_df, how='outer', on="radius")[['radius', 'eigen-index', 'random-index', 'E_loss', 'E_loss_rnd']]
    merge_df['E_loss_ratio']= merge_df['E_loss']/(merge_df['E_loss_rnd'])
    merge_df.loc[(merge_df['E_loss'].isna()) & (merge_df['E_loss_rnd'].isna()),'E_loss_ratio'] = 1
    merge_df.loc[(merge_df['E_loss'].notna()) & (merge_df['E_loss_rnd'].isna()),'E_loss_ratio'] = 0
    merge_df = merge_df.groupby(['radius','eigen-index']).agg({'E_loss':'mean', 'E_loss_ratio':'mean'}).reset_index()
    res_df = merge_df[['radius','eigen-index','E_loss','E_loss_ratio']]
    for i in range(L.size(0)):
        res_df.loc[res_df['eigen-index']==i,'eigen-value'] = L[i].item()
    return res_df


columns=['radius','eigen-index','E_loss','E_loss_ratio',
         'P','seed','task','dataset','model','lr', 'train']
perturbations_df = pd.DataFrame(columns=columns)

args = parse_args()
torch.set_default_dtype(torch.float32)
#instantiating model and dataset 
dataset = get_dataset(args)
args.batch_size = dataset.get_batch_size()
backbone = dataset.get_backbone()
loss = dataset.get_loss()
model = get_model(args, backbone, loss, dataset.get_transform())
# count number of parameters 
P = parameters_to_vector(model.parameters()).detach().shape[0]

base_path_chkpts = base_path_checkpoints() +  dataset.SETTING + "/" + dataset.NAME + "/" + model.NAME 
base_path_logs = base_path() + "/results/" + dataset.SETTING + "/" + dataset.NAME + "/" + model.NAME

print(f"Starting perturbation analysis of {model.name}")

for t in tqdm(range(dataset.N_TASKS)):
    train_loader, test_loader = dataset.get_data_loaders()
    # load checkpoint for the given task 
    chkpt_name = f"/task{t}_{model.name}.pt"
    chkpt = model.load_checkpoint(path=base_path_chkpts+chkpt_name, device=model.device)
    model.net.load_state_dict(chkpt['state_dict'])
    model.net.eval(); model.net.to(model.device)
    # perturbation analysis for train 
    print("Perturbations on train.")
    res_df = do_perturbation_analysis(model, train_loader, num_eigenthings=10)
    res_df[['seed','task','P','dataset','model','lr', 'train']] = [args.seed, t, P, dataset.NAME, model.NAME, args.lr, True]
    perturbations_df = pd.concat([res_df, perturbations_df])
    # perturbation analysis for test 
    chkpt = model.load_checkpoint(path=base_path_chkpts+chkpt_name, device=model.device)
    model.net.load_state_dict(chkpt['state_dict'])
    model.net.eval(); model.net.to(model.device)
    print("Perturbations on test.")
    res_df = do_perturbation_analysis(model, test_loader, num_eigenthings=10, test=True)
    res_df[['seed','task','P','dataset','model','lr', 'train']] = [args.seed, t, P, dataset.NAME, model.NAME, args.lr, False]
    perturbations_df = pd.concat([res_df, perturbations_df])
    # saving results 
    logs_name = f"ptb_{model.name}.csv"
    perturbations_df.to_csv(os.path.join(base_path_logs,logs_name))

print("Perturbation analysis completed.")
