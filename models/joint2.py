# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
from datasets.utils.validation import ValidationDataset
from torch.optim import SGD
from torchvision import transforms
from torch.utils.data import ConcatDataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from utils.status import progress_bar

# making the logic of the joint model data loading easier using the 'ConcatDataset' function from pytorch

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Joint2(ContinualModel):
    NAME = 'joint2'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint2, self).__init__(backbone, loss, args, transform)
        self.old_datasets = []
        self.current_task = 0

    def end_task(self, dataset):

        self.old_datasets.append(dataset.train_loader.dataset)

        # # for non-incremental joint training
        #if len(dataset.test_loaders) != dataset.N_TASKS: return
        
        scheduler = dataset.get_scheduler(self, self.args)
        
        self.current_task += 1
        
        # reinit network
        self.net = dataset.get_backbone()
        self.net.to(self.device)
        self.net.train()
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
    
        # prepare dataloader
        joint_dataset = ConcatDataset(self.old_datasets)
        loader = torch.utils.data.DataLoader(joint_dataset, batch_size=self.args.batch_size, shuffle=True)

        # train
        for e in range(self.args.n_epochs):
            for i, batch in enumerate(loader):
                inputs, labels, _ = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss(outputs, labels.long())
                loss.backward()
                self.opt.step()
                progress_bar(i, len(loader), e, 'J', loss.item())

            if scheduler is not None:
                scheduler.step()

    def observe(self, inputs, labels, not_aug_inputs):
        return 0
