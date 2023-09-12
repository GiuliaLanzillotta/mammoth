""" Script implementing different distillation losses."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18
import torchvision.transforms as transforms


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


def inner_distillation(student_activations, teacher_activations):
    """ Distillation comparing the outputs of all network's layers. """
    loss = 0.
    for n, act_s in student_activations.items(): 
        act_t = teacher_activations[n] # it will throw an error if the two networks are different
        loss += F.mse_loss(act_s, act_t)
    return loss

