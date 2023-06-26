# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.nn import functional as F

import math

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--probabilities', default=False, action="store_true",
                        help='Distillation at the probabilities level.')
    parser.add_argument('--temp', type=float,  default=10, 
                        help='Softmax temperature for probability-based distillation.')
    return parser


class TDer(ContinualModel):
    NAME = 'tder'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(TDer, self).__init__(backbone, loss, args, transform)
        self.use_prob = self.args.probabilities
        self.current_task=0
        self.buffer = Buffer(self.args.buffer_size, self.device)


    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if self.current_task > 0: # no replay while learning the first task.
            buf_inputs, buf_logits, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            if self.use_prob: 
                #TODO: customise activation function
                buf_outputs = F.softmax(buf_outputs); buf_logits = F.softmax(buf_logits/self.args.temp)
                loss += self.args.alpha * F.cross_entropy(buf_outputs, buf_logits)
            else: loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

        loss.backward()
        self.opt.step()

        return loss.item()
    
    def end_task(self, dataset): 
        """ Pass through the dataset once more to store the logits at the end of training. """
        status = self.net.training
        self.net.eval()

        for i, data in enumerate(dataset.train_loader):
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            not_aug_inputs = not_aug_inputs.to(self.device)
            with torch.no_grad():
                outputs = self.net(inputs)
            task_labels = torch.ones(inputs.shape[0])*self.current_task
            self.buffer.add_data(examples=not_aug_inputs, 
                                logits=outputs.data,
                                task_labels=task_labels)
            
        self.net.train(status)
        self.current_task +=1
