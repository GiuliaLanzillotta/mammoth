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
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.status import progress_bar


from torch.nn import functional as F

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training with distillation.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--buffer', default=False, action='store_true',
                        help='Whether to train the joint model on a buffer.')
    parser.add_argument('--task_buffer', default=False, action='store_true',
                        help='Whether to store logits in the buffer at the end of training.')
    parser.add_argument('--alpha', type=float, default=0.5, required=True,
                        help='The weight of labels vs logits in the distillation loss (when alpha=1 only true labels are used)')
    parser.add_argument('--reset_fl', default=False, action='store_true',
                        help='Whether to reset the fast learner at the beginning of each new task.')
    return parser


class JointDistill(ContinualModel):
    """ Joint training with distillation from a fast-learning network."""
    NAME = 'joint_distill'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(JointDistill, self).__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.fast_learner = None
        if self.args.buffer:
            self.buffer = Buffer(self.args.buffer_size, self.device)
        else: self.old_datasets = []

    def begin_task(self, dataset):
        """ Reset the 'fast' network to train on the dataset from scratch """
        #TODO: choose whether to re-initialise it for each task
        if self.args.reset_fl or self.current_task==0:
            self.fast_learner = dataset.get_backbone()
            self.fast_learner.to(self.device)
            self.fast_learner.train()
            self.fast_opt = SGD(self.fast_learner.parameters(), lr=self.args.lr)


    def observe(self, inputs, labels, not_aug_inputs):
        """Fast learner module assimilates the dataset"""

        self.fast_opt.zero_grad()
        outputs = self.fast_learner(inputs)

        loss = self.loss(outputs, labels)
        loss.backward()
        self.fast_opt.step()

        if self.args.buffer and not self.args.task_buffer:
            self.buffer.add_data(examples=not_aug_inputs, # augmentations applied by the buffer 
                             logits=outputs.data, labels=labels)

        return loss.item()


    def slow_learn_buffer(self, dataset):
        
        self.current_task += 1
        # reinit network
        self.net = dataset.get_backbone()
        self.net.to(self.device)
        self.net.train()
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        alpha = self.args.alpha
        scheduler = dataset.get_scheduler(self, self.args)

        bs = self.args.minibatch_size
        for e in range(self.args.n_epochs):
            for i in range(int(math.ceil(self.args.buffer_size / bs))):
                inputs, labels, logits = self.buffer.get_data(bs, transform=self.transform)
                inputs, labels, logits = inputs.to(self.device), logits.to(self.device), labels.to(self.device)
                self.opt.zero_grad()
                outputs = self.net(inputs)
                logits_loss = F.mse_loss(outputs, logits)
                labels_loss = self.loss(outputs, labels)
                loss = alpha*labels_loss + (1-alpha)*logits_loss
                loss.backward()
                self.opt.step()
                progress_bar(i, int(math.ceil(self.args.buffer_size / bs)), e, 'JD', loss.item())

            if scheduler is not None:
                scheduler.step()
    
    def slow_learn_full(self, dataset):
        """Joint training from all previous datasets..."""

        # collect all the logits for the current dataset from the fast learning model 
        logits = []
        with torch.no_grad():
            for i in range(0, dataset.train_loader.dataset.data.shape[0], self.args.batch_size):
                inputs = torch.stack([dataset.train_loader.dataset.__getitem__(j)[2]
                                        for j in range(i, min(i + self.args.batch_size,
                                                        len(dataset.train_loader.dataset)))])
                log = self.fast_learner(inputs.to(self.device)).cpu()
                logits.append(log)
        setattr(dataset.train_loader.dataset, 'logits', torch.cat(logits))
    
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
        alpha = self.args.alpha
    
        # prepare dataloader
        joint_dataset = ConcatDataset(self.old_datasets)
        loader = torch.utils.data.DataLoader(joint_dataset, batch_size=self.args.batch_size, shuffle=True)

        # train
        for e in range(self.args.n_epochs):
            for i, batch in enumerate(loader):
                inputs, labels, _, logits = batch
                inputs, logits, labels = inputs.to(self.device), logits.to(self.device), labels.to(self.device)

                self.opt.zero_grad()
                outputs = self.net(inputs)
                logits_loss = F.mse_loss(outputs, logits)
                labels_loss = self.loss(outputs, labels)
                loss = alpha*labels_loss + (1-alpha)*logits_loss
                loss.backward()
                self.opt.step()
                progress_bar(i, len(loader), e, 'J', loss.item())

            if scheduler is not None:
                scheduler.step()
                
 

    def end_task(self, dataset):
        """The full network learns the task now using the fast network outputs."""
        if self.args.buffer and self.args.task_buffer: 
            # we fill up the buffer at the end of the task training (using the trained fast network)
            status = self.fast_learner.training
            self.fast_learner.eval()

            for i, data in enumerate(dataset.train_loader):
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                not_aug_inputs = not_aug_inputs.to(self.device)
                with torch.no_grad():
                    outputs = self.fast_learner(inputs)
                self.buffer.add_data(examples=not_aug_inputs, 
                                    logits=outputs.data, labels=labels)
                
            self.fast_learner.train(status)

        if self.args.buffer: self.slow_learn_buffer(dataset)
        else: self.slow_learn_full(dataset)
        