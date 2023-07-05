# Giulia Lanzillotta . 04.07.2023
# Imagenet offline training experiment script (no continual structure)


import importlib
import math
import os
import socket
import sys

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
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageNet
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights
import torchvision.transforms as transforms

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
              '1000':{
                  'alpha':0.0  
              }  
        }


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    path = base_path() + "/chkpts" + "/" + "imagenet" + "/" + "resnet50/"
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(filename, path+'model_best.pth.tar')

def load_checkpoint(best=False, filename='checkpoint.pth.tar'):
    path = base_path() + "/chkpts" + "/" + "imagenet" + "/" + "resnet50/"
    if best: filepath = path + 'model_best.pth.tar'
    else: filepath = path + filename
    if os.path.exists(filepath):
          print(f"Loading existing checkpoint {filepath}")
          checkpoint = torch.load(filepath)
          return checkpoint
    return None 

def evaluate(model, val_loader, device):
    status = model.training
    model.eval()
    progress_bar = ProgressBar(verbose=not args.non_verbose)
    correct, total = 0.0, 0.0
    for i,data in enumerate(val_loader):
        with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                progress_bar.prog(i, len(val_loader), 'validation', '-', (correct/total)*100)
                        
        acc=(correct / total) * 100
    model.train(status)
    return acc

def parse_args(buffer=False):
    torch.set_num_threads(4)
    parser = ArgumentParser(description='script-experiment', allow_abbrev=False)
    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])
    parser.add_argument('--lr', type=float, required=False,
                        help='Learning rate.')
    parser.add_argument('--checkpoints', action='store_true', help='Storing a checkpoint at every epoch. Loads a checkpoint if present.')
    parser.add_argument('--pretrained', action='store_true', help='Using a pre-trained network instead of training one.')
    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')
    
    add_management_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default=0.5, required=True,
                        help='The weight of labels vs logits in the distillation loss (when alpha=1 only true labels are used)')
    parser.add_argument('--noisy_buffer', default=False, action='store_true',
                        help='Whether to store logits in the buffer at the end of training.')
    args = parser.parse_known_args()[0]

    if buffer:
        best = buffer_args[args.buffer_size] #TODO
        to_parse = ['--' + k + '=' + str(v) for k, v in best.items()] + sys.argv[1:] # this way the argv args can override the best args
        args = parser.parse_args(to_parse)
        
    else:
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


args = parse_args()
# Add uuid, timestamp and hostname for logging
args.conf_jobnum = str(uuid.uuid4())
args.conf_timestamp = str(datetime.datetime.now())
args.conf_host = socket.gethostname()

# dataset -> imagenet
imagenet_root = "/media/hofmann-scratch/data/imagenet"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# Resizing
# interpolation='bilinear', 
# val_resize_size=256, 
# val_crop_size=224, 
# train_crop_size=224,

TRANSFORM = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
train_dataset = ImageNet(root=imagenet_root, split="train", transform=TRANSFORM)
val_dataset = ImageNet(root=imagenet_root, split="val", transform=TRANSFORM)
# filling in default hyperparameters
if args.n_epochs is None: args.n_epochs = 90
if args.batch_size is None: args.batch_size = 32
if args.lr is None: args.lr = 0.1

# initialising the model
weights = None
if args.pretrained: 
      print("Loading pretrained weights...")
      weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)

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
device = get_device(args.gpuid)
model.to(device)
progress_bar = ProgressBar(verbose=not args.non_verbose)


print(file=sys.stderr)
train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

buffer = Buffer(args.buffer_size, device) # reservoir sampling during learning

if not args.pretrained:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        results = []
        best_acc = 0.
        start_epoch = 0


        if args.checkpoints: 
                chkpt_name = str.join("-",["offline", "imagenet", "resnet50", str(args.seed)])
                checkpoint = load_checkpoint(best=True, filename=chkpt_name) #TODO: switch best off
                model.load_state_dict(checkpoint['state_dict'])
                model.to(device)
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']


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
                val_acc = evaluate(model, val_loader, device)
                results.append(val_acc)

                # best val accuracy -> selection bias on the validation set
                is_best = val_acc > best_acc
                best_acc = max(val_acc, best_acc)

                print('\Train accuracy : {} %'.format(
                round(train_acc, 2)), file=sys.stderr)
                
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

val_acc = evaluate(model, val_loader, device)          
df = {'final_val_acc_D':val_acc}
wandb.log(df)


if not args.noisy_buffer:
      #fill-up the buffer now (when the standard model is trained)
      status = model.training
      model.eval()
      for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                buffer.add_data(examples=inputs, logits=outputs.data, labels=labels)      
      model.train(status)



args = parse_args(buffer=True)
# dumping everything into a log file
path = base_path() + "/results" + "/" + "imagenet" + "/" + "resnet50" + "/logs.pyd"
with open(path, 'a') as f:
        f.write(str(vars(args)) + '\n')



print("Starting buffer training ... ")
if args.n_epochs is None: args.n_epochs = 90
if args.batch_size is None: args.batch_size = 32
if args.lr is None: args.lr = 0.1
if args.optim_wd is None: args.optim_wd = 1e-4
if args.optim_mom is None: args.optim_mom = 0.9
# re-initialise model 
buffer_model = resnet50(weights=None)

buffer_model.train()
train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
optimizer = torch.optim.SGD(buffer_model.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

results = []
alpha = args.alpha
for e in range(args.n_epochs):
        avg_loss = 0.0
        correct, total = 0.0, 0.0
        for i in range(int(math.ceil(args.buffer_size / args.batch_size))):
                if args.debug_mode and i > 3: # only 3 batches in debug mode
                        break
                inputs, labels, logits = buffer.get_data(args.batch_size)
                inputs, logits, labels = inputs.to(device), logits.to(device), labels.to(device)
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
                progress_bar(i, int(math.ceil(args.buffer_size / args.batch_size)), e, 'S', loss.item())
                avg_loss += loss
        
        avg_loss = avg_loss/i
        if scheduler is not None:
                scheduler.step()
        
        train_acc = (correct/total) * 100
        val_acc = evaluate(model, val_loader, device)
        results.append(val_acc)

        print('\Train accuracy : {} %'.format(
            round(train_acc, 2)), file=sys.stderr)
        
        df = {'epoch_loss_S':avg_loss/len(train_loader),
              'epoch_train_acc_S':train_acc,
              'epoch_val_acc_S':val_acc}
        wandb.log(df)

if not args.nowand:
        wandb.finish()


# https://github.com/pytorch/examples/blob/main/imagenet/main.py 
# https://pytorch.org/vision/0.8/datasets.html#imagenet
# https://www.image-net.org/about.php 