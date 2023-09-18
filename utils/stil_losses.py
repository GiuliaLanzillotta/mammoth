""" Script implementing different distillation losses."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18
import torchvision.transforms as transforms
from utils.nets import DictionaryNet


def vanilla_distillation(student_out, teacher_out): 
    """ Classic distillation comparing the last layer activations in the networks. 
        student_out: BxO tensor
        teacher_out: BxO tensor 
    """
    return F.mse_loss(student_out, teacher_out)

def topK_distillation(student_out, teacher_out, K):
    """ Distillation comparing the top-K last layer activations, 
    where the ordering is based on the [teacher's network predictions ?].
    """
    _, topk_idx = torch.sort(teacher_out.clone().detach(), descending=True)
    teach_topk = torch.gather(teacher_out, dim = 1, index = topk_idx)[:,:K]
    stud_topk = torch.gather(student_out, dim = 1, index = topk_idx)[:,:K]
    return F.mse_loss(stud_topk, teach_topk)


def inner_distillation(student_net:DictionaryNet, teacher_activations, x):
    """ Distillation comparing the outputs of all network's layers. """
    loss = 0.
    count = 0
    for n, act_s in teacher_activations.items(): 
        if n=='fc': x = torch.flatten(x, 1)
        act_t = teacher_activations[n] # it will throw an error if the two networks are different
        act_s = student_net(x, name=n)
        loss += F.mse_loss(act_s, act_t)
        count+=1
        x = act_t
    return loss/count #average across depth

def topbottomK_distillation(student_out, teacher_out, K):
    """ Distillation comparing the top-K and bottom-K last layer activations, 
    where the ordering is based on the teacher's network predictions.
    """
    k = K//2 # we use
    _, ordered_idx = torch.sort(teacher_out.clone().detach(), descending=True)
    teach_topk = torch.gather(teacher_out, dim = 1, index = ordered_idx)[:,:k]
    teach_bottomk = torch.gather(teacher_out, dim = 1, index = ordered_idx)[:,-k:]
    stud_topk = torch.gather(student_out, dim = 1, index = ordered_idx)[:,:k]
    stud_bottomk = torch.gather(student_out, dim = 1, index = ordered_idx)[:,-k:]
    return 0.5*F.mse_loss(stud_topk, teach_topk) + 0.5*F.mse_loss(stud_bottomk, teach_bottomk)


def randomK_distillation(student_out, teacher_out, K):
    """ Distillation comparing a random K subset of the last layer activations, 
    where the ordering is based on the teacher's network predictions.
    """
    random_idx = torch.multinomial(torch.ones_like(teacher_out), K)
    teach_randk = torch.gather(teacher_out, dim = 1, index = random_idx)
    stud_randk = torch.gather(student_out, dim = 1, index = random_idx)
    return F.mse_loss(stud_randk, teach_randk)

