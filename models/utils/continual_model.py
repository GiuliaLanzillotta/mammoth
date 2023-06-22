# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from contextlib import suppress
from typing import List

import torch
import torch.nn as nn
from torch.optim import SGD

from utils.conf import get_device
from utils.magic import persistent_locals
from utils.status import ProgressBar

from torch.utils.data import DataLoader


with suppress(ImportError):
    import wandb


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str]

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.device = get_device(args.gpuid)

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        if 'wandb' in sys.modules and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            self.autolog_wandb(pl.locals)
        else:
            ret = self.observe(*args, **kwargs)
        return ret

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

    # def train_on_task(self, train_loader:DataLoader, 
    #                   scheduler:torch.optim.lr_scheduler._LRScheduler, 
    #                   progress_bar:ProgressBar): 
    #     """
    #     Compute a full training cycle through a task. 
    #     :param train_loader: training data loader
    #     :param 
    #     """
    #     for epoch in range(self.args.n_epochs):
    #         if self.args.model == 'joint':
    #             continue
    #         for i, data in enumerate(train_loader):
    #             if self.args.debug_mode and i > 3: # only 3 batches in debug mode
    #                 break
    #             if hasattr(train_loader.dataset, 'logits'):
    #                 inputs, labels, not_aug_inputs, logits = data
    #                 inputs = inputs.to(self.device)
    #                 labels = labels.to(self.device)
    #                 not_aug_inputs = not_aug_inputs.to(self.device)
    #                 logits = logits.to(self.device)
    #                 loss = self.meta_observe(inputs, labels, not_aug_inputs, logits)
    #             else:
    #                 inputs, labels, not_aug_inputs = data
    #                 inputs, labels = inputs.to(self.device), labels.to(self.device)
    #                 not_aug_inputs = not_aug_inputs.to(self.device)
    #                 loss = self.meta_observe(inputs, labels, not_aug_inputs)
    #             assert not math.isnan(loss)
    #             progress_bar.prog(i, len(train_loader), epoch, t, loss)

    #         if scheduler is not None:
    #             scheduler.step()

    def autolog_wandb(self, locals):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            wandb.log({k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                      for k, v in locals.items() if k.startswith('_wandb_') or k.startswith('loss')})
