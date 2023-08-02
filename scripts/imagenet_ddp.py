# Giulia Lanzillotta . 04.07.2023
# Imagenet offline training experiment script (no continual structure)

#Resources
# https://github.com/pytorch/examples/blob/main/imagenet/main.py 
# https://pytorch.org/vision/0.8/datasets.html#imagenet
# https://www.image-net.org/about.php 


import importlib
import math
import os
import socket
import sys
import time


os.putenv("MKL_SERVICE_FORCE_INTEL", "1") #CHECK
os.putenv("NPY_MKL_FORCE_INTEL", "1")

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')
sys.path.append(mammoth_path + '/utils')


import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch

from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import NAMES as DATASET_NAMES
from datasets import ContinualDataset, get_dataset
from models import get_all_models, get_model


import shutil
from utils.args import add_management_args, add_rehearsal_args
from utils.conf import set_random_seed, get_device
from utils.loggers import *
from utils.status import ProgressBar
from utils.buffer import Buffer

try:
    import wandb
except ImportError:
    wandb = None


buffer_args = {
                'n_epochs':90,
                'batch_size':256, #TODO: tune
                'validate_subset':5000,
                'lr':0.1
                }


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    path = base_path() + "/chkpts" + "/" + "imagenet" + "/" + "resnet50/"
    if not os.path.exists(path): os.makedirs(path)
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(filename, path+'model_best.pth.tar')

def load_checkpoint(map_location, best=False, filename='checkpoint.pth.tar'):
    path = base_path() + "/chkpts" + "/" + "imagenet" + "/" + "resnet50/"
    if best: filepath = path + 'model_best.pth.tar'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath, map_location=map_location)
        #   if filename=='checkpoint_90.pth.tar': # modify Sidak's checkpoint
        #         new_state_dict = {k.replace('module.','',1):v for (k,v) in checkpoint['state_dict'].items()}
        #         checkpoint['state_dict'] = new_state_dict
          return checkpoint
    return None 

def evaluate(model, val_loader, device, num_samples=-1):
    status = model.training
    model.eval()
    if num_samples >0: 
          # we select a subset of the validation dataset to validate on 
          # note: a different random sample is used every time
          random_indices = np.random.choice(range(len(val_loader.dataset)), size=num_samples, replace=False)
          _subset = Subset(val_loader.dataset, random_indices)
          val_loader =  DataLoader(_subset, 
                                   batch_size=val_loader.batch_size, 
                                   shuffle=False, num_workers=4, pin_memory=False)
    correct, total = 0.0, 0.0
    for i,data in enumerate(val_loader):
        with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                        
        acc=(correct / total) * 100
    model.train(status)
    return acc

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def parse_args(buffer=False):
    torch.set_num_threads(4)
    parser = ArgumentParser(description='script-experiment', allow_abbrev=False)
    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate.')
    parser.add_argument('--checkpoints', action='store_true', help='Storing a checkpoint at every epoch. Loads a checkpoint if present.')
    parser.add_argument('--pretrained', action='store_true', help='Using a pre-trained network instead of training one.')
    parser.add_argument('--optim_wd', type=float, default=1e-4, help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.9, help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0, help='optimizer nesterov momentum.')
    parser.add_argument('--n_epochs', type=int, default=90, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default = 256, help='Batch size.')
    parser.add_argument('--validate_subset', type=int, default=-1, 
                        help='If positive, allows validating on random subsets of the validation dataset during training.')
    
    add_management_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--alpha', type=float, default=0.5, required=True,
                        help='The weight of labels vs logits in the distillation loss (when alpha=1 only true labels are used)')
    parser.add_argument('--noisy_buffer', default=False, action='store_true',
                        help='Whether to store logits in the buffer at the end of training.')
    args = parser.parse_known_args()[0]

    if buffer:
        best = buffer_args
        to_parse = ['--' + k + '=' + str(v) for k, v in best.items()] + sys.argv[1:] # this way the argv args can override the best args
        args = parser.parse_args(to_parse)
        
    else:
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def train_teacher_and_student(rank, world_size, args):
        setup(rank, world_size)
        # initialising the model
        weights = None
        if args.pretrained: 
                print("Loading pretrained weights...")
                weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights).to(args.gpus_id[rank])
        model = DDP(model, device_ids=[args.gpus_id[rank]])

        device = args.gpus_id[rank] 
        progress_bar = ProgressBar(verbose=not args.non_verbose)
        buffer = Buffer(args.buffer_size, device) # reservoir sampling during learning


        if not args.pretrained:
                model.train()
                optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=args.lr, 
                                    weight_decay=args.optim_wd, 
                                    momentum=args.optim_mom)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
                results = []
                best_acc = 0.
                start_epoch = 0


                if args.checkpoints: 
                        chkpt_name = f"checkpoint_90.pth.tar" #sidak's checkpoint
                        checkpoint = load_checkpoint(map_location={'cuda:%d' % 0: 'cuda:%d' % rank}, 
                                                     best=False, filename=chkpt_name) 
                        model.load_state_dict(checkpoint['state_dict'])
                        model.to(device)
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        scheduler.load_state_dict(checkpoint['scheduler'])
                        start_epoch = checkpoint['epoch']
                        best_acc1 = checkpoint['best_acc1']


                for epoch in range(start_epoch, args.n_epochs):
                        avg_loss = 0.0
                        correct, total = 0.0, 0.0
                        for i, data in enumerate(train_loader):
                                if args.debug_mode and i > 3: # only 3 batches in debug mode
                                        break
                                inputs, labels = data
                                inputs, labels = inputs.to(device), labels.to(device)
                                outputs = model(inputs)
                                _, pred = torch.max(outputs.data, 1)
                                correct += torch.sum(pred == labels).item()
                                total += labels.shape[0]
                                loss = F.cross_entropy(outputs, labels) #TODO: maybe MSE?
                                loss.backward()
                                optimizer.step()

                                assert not math.isnan(loss)
                                progress_bar.prog(i, len(train_loader), epoch, 'D', loss)
                                avg_loss += loss

                                if args.noisy_buffer:             
                                        buffer.add_data(examples=inputs, logits=outputs.data, labels=labels)

                        if scheduler is not None:
                                scheduler.step()
                        
                        train_acc = correct/total * 100
                        val_acc = evaluate(model, val_loader, device, num_samples=args.validate_subset)
                        results.append(val_acc)

                        # best val accuracy -> selection bias on the validation set
                        is_best = val_acc > best_acc
                        best_acc = max(val_acc, best_acc)

                        print('\Train accuracy : {} %'.format(round(train_acc, 2)), file=sys.stderr)
                        print('\Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
                        
                        df = {'epoch_loss_D':avg_loss/len(train_loader),
                                'epoch_train_acc_D':train_acc,
                                'epoch_val_acc_D':val_acc}
                        wandb.log(df)


                if args.checkpoints: 
                        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_acc': best_acc,
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict()
                        }, is_best, filename=chkpt_name)

                #val_acc = evaluate(model, val_loader, device)          
                final_val_acc_D = checkpoint['best_acc1']
                df = {'final_val_acc_D':final_val_acc_D}
                wandb.log(df)
        

        cleanup()



def run_demo(fn, world_size, args):
    world_size = len(args.gpus_id)
    mp.spawn(fn,
             args=(world_size,args,),
             nprocs=world_size,
             join=True)



args = parse_args()
# Add uuid, timestamp and hostname for logging
args.conf_jobnum = str(uuid.uuid4())
args.conf_timestamp = str(datetime.datetime.now())
args.conf_host = socket.gethostname()

# dataset -> imagenet
imagenet_root = "/local/home/stuff/imagenet/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ])
inference_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])

train_dataset = ImageFolder(imagenet_root+'train', train_transform)
val_dataset = ImageFolder(imagenet_root+'val', inference_transform)

print(file=sys.stderr)
train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)




#TODO: data parallel switch
setproctitle.setproctitle('{}_{}_{}'.format("resnet50", args.buffer_size if 'buffer_size' in args else 0, "imagenet"))

# start the training 
print(args)
if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if args.wandb_name is None: 
                name = str.join("-",["offline", "imagenet", "resnet50", args.conf_timestamp])
        else: name = args.wandb_name
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
                        name=name, notes=args.notes, config=vars(args)) 
        args.wandb_url = wandb.run.get_url()





print(f"Randomly drawing {args.buffer_size} samples")
model.eval() # set the main model to evaluation
all_indices = set(range(len(train_dataset)))
random_indices = np.random.choice(list(all_indices), 
                                size=args.buffer_size, replace=False)
left_out_indices = all_indices.difference(set(random_indices.flatten()))
train_subset = Subset(train_dataset, random_indices)
buffer_loader =  DataLoader(train_subset, 
                                batch_size=args.batch_size, 
                                shuffle=True, 
                                num_workers=4, 
                                pin_memory=True)
train_leftout_subset = Subset(train_dataset, list(left_out_indices))
train_leftout_loader = DataLoader(train_leftout_subset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=False)


args = parse_args(buffer=True)
experiment_log = vars(args)
experiment_log['final_val_acc_D'] = final_val_acc_D.detach().cpu().numpy()



print("Starting buffer training ... ")
start = time.time()
# re-initialise model 
buffer_model = resnet50(weights=None)
if args.distributed=='dp': 
      print(f"Parallelising buffer training on {len(args.gpus_id)} GPUs.")
      buffer_model = torch.nn.DataParallel(buffer_model, device_ids=args.gpus_id).cuda()
buffer_model.to(device)
buffer_model.train()
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                        shuffle=False, num_workers=3, pin_memory=True)

optimizer = torch.optim.SGD(buffer_model.parameters(), 
                            lr=args.lr, 
                            weight_decay=args.optim_wd, 
                            momentum=args.optim_mom)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
results = []
alpha = args.alpha
for e in range(args.n_epochs):
        if args.debug_mode and e > 3: # only 3 batches in debug mode
                break
        avg_loss = 0.0
        correct, total = 0.0, 0.0
        for i, data in enumerate(buffer_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad(): logits = model(inputs)
                optimizer.zero_grad()
                outputs = buffer_model(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                logits_loss = F.mse_loss(outputs, logits)
                labels_loss = F.cross_entropy(outputs, labels)
                loss = alpha*labels_loss + (1-alpha)*logits_loss
                loss.backward()
                optimizer.step()

                assert not math.isnan(loss)
                progress_bar.prog(i, len(buffer_loader), e, 'S', loss.item())
                avg_loss += loss
        
        avg_loss = avg_loss/i
        if scheduler is not None:
                scheduler.step()
        
        train_acc = (correct/total) * 100
        train_leftout_acc = evaluate(buffer_model, train_leftout_loader, device, num_samples=args.validate_subset)
        val_acc = evaluate(buffer_model, val_loader, device, num_samples=args.validate_subset)
        results.append(val_acc)

        print('\nTrain accuracy : {} %'.format(round(train_acc, 2)), file=sys.stderr)
        print('Train left-out accuracy : {} %'.format(round(train_leftout_acc, 2)), file=sys.stderr)
        print('Val accuracy : {} %'.format(round(val_acc, 2)), file=sys.stderr)
        
        df = {'epoch_loss_S':avg_loss,
              'epoch_train_acc_S':train_acc,
              'epoch_train_leftout_acc_S':train_leftout_acc,
              'epoch_val_acc_S':val_acc}
        wandb.log(df)


print("Training completed. Full evaluation and logging...")
end = time.time()

experiment_log['buffer_train_time'] = end-start
experiment_log['final_train_acc_S'] = train_acc
train_leftout_acc = evaluate(buffer_model, train_leftout_loader, device)
val_acc = evaluate(buffer_model, val_loader, device)
experiment_log['final_train_leftout_acc_S'] = train_leftout_acc
experiment_log['final_val_acc_S'] = val_acc


if not args.nowand:
        wandb.finish()


# dumping everything into a log file
path = base_path() + "results" + "/" + "imagenet" + "/" + "resnet50" 
if not os.path.exists(path): os.makedirs(path)
with open(path+ "/logs.pyd", 'a') as f:
        f.write(str(experiment_log) + '\n')