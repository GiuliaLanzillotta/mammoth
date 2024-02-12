# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the gem_license file in the root of this source tree.

import random
import numpy as np
import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Orthogonal Gradient Descent.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--mode', type=str, default='all', choices=['all', 'ave', 'gtl'],
                        help='OGD approximation type.')
    return parser


def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def orthonormalize(vectors, normalize=True, start_idx=1):
    """Applies orthonormalisation to the matrix 'vectors', which should be a torch tensor,
    starting from the 'start_idx' column. Notice that the operations are not in place."""

    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'

    O = torch.zeros_like(vectors) #orthogonal matrix 

   
    if start_idx==0: start_idx=1
    O[:,:start_idx] = vectors[:,:start_idx]
    if normalize:
        O[:, :start_idx] /= torch.linalg.norm(vectors[:, :start_idx], ord=2, dim=0).view(1,start_idx)

    total = start_idx # counting the total number of directions used

    for i in range(start_idx, vectors.size(1)): # go through the remaining columns
        with torch.no_grad():
            vector = vectors[:, i] # vector to orthogonalise
            V = O[:, :total] # basis wrt which orthogonalise
            PV_vector = torch.mv(V, torch.mv(V.t(), vector)) # matrix-vector product -> projection
            residual = vector - PV_vector
            residual_norm = torch.linalg.vector_norm(residual, ord=2)
            if torch.allclose(residual_norm, torch.Tensor([0.]).to(residual_norm.device)): continue
            O[:,total] = residual
            if normalize: O[:, total] /= residual_norm
            total+=1 
    
    O = O[:,:total]
    #final check of orthogonality
    print(torch.dist(O.T @ O, torch.eye(O.size(1)).to(O.device)))

    return O

# def orthonormalize(vectors, normalize=True):
#     """Applies orthonormalisation to the matrix 'vectors', which should be a torch tensor,
#     using a qr decomposition. Notice that the operations are not in place."""

#     assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'

#     O, _ = torch.linalg.qr(vectors, mode="reduced")
#     #final check of orthogonality

#     #assert torch.allclose(torch.mm(O.T,O), torch.eye(O.size(1)).to(O.device),atol=1e-06), "The orthogonalisation failed."
#     print(torch.dist(O.T @ O, torch.eye(O.size(1)).to(O.device)))
#     return O

class Ogd(ContinualModel):
    NAME = 'ogd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Ogd, self).__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.buffer = Buffer(self.args.buffer_size, self.device)

        # Allocate temporary synaptic memory
        self.grad_dims = []
        for pp in self.parameters():
            self.grad_dims.append(pp.data.numel())

        self.grads_cs = []
        self.grads_da = torch.zeros(np.sum(self.grad_dims)).to(self.device)
        self.grads_mat = None


    def collect_grads(self, dataset, n):
        """ Collects gradient of n random samples of the dataset"""
        indices = random.sample(list(range(len(dataset.train_loader.dataset))), n) # samples indices
        samples = torch.utils.data.Subset(dataset.train_loader.dataset, indices) #creating a dataset with these samples
        dataloader = torch.utils.data.DataLoader(samples, shuffle=False, num_workers=2, batch_size=1)

        gradients = []

        status = self.net.training
        self.net.train()  # we don't need gradients
        
        for inputs, labels, not_aug_inputs in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # now compute the grad on the current data
            self.opt.zero_grad()
            outputs = self.forward(inputs)
            if self.args.mode == "gtl": # select target class gradient
                outputs_t = outputs.gather(dim=1, index=labels.view(-1,1)) 
                outputs_t.backward()
            elif self.args.mode == "ave": # average all gradients
                outputs_ave = outputs.mean(dim=1)
                outputs_ave.backward()
            elif self.args.mode == "all": #TODO:fix 
                for c in range(outputs.size(1)): # accumulate all gradients
                    outputs_c = outputs.gather(dim=1, index=c) 
                    outputs_c.backward()
            else: raise NotImplementedError(self.args.mode+"-OGD not implemented.")

            # copy gradient 
            store_grad(self.parameters, self.grads_da, self.grad_dims)
            gradients.append(self.grads_da.unsqueeze(1).detach()) # column vectors


        gradients = torch.hstack(gradients) # size (B,N)
        self.net.train(mode=status)
        return gradients
        

    def end_task(self, dataset):
        self.current_task += 1

        # update the gradient matrix 
        samples_per_task = self.args.buffer_size // dataset.N_TASKS
        gradients = self.collect_grads(dataset, samples_per_task)
        m = 0
        if self.grads_mat is not None: 
            m = self.grads_mat.size(1)
            gradients = torch.hstack([self.grads_mat, gradients]).detach()
        self.grads_mat = orthonormalize(gradients, normalize=True, start_idx=m)
        # adding extra check to avoid gradient explosion
        self.grads_mat /= torch.linalg.norm(self.grads_mat)
        print(f"{self.grads_mat.size(1)} gradients stored.")


    def observe(self, inputs, labels, not_aug_inputs):

        # now compute the grad on the current data
        self.opt.zero_grad()
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        if self.grads_mat is not None:
            # check if gradient violates constraints
            # copy gradient
            store_grad(self.parameters, self.grads_da, self.grad_dims)

            dot_prod =  torch.mv(self.grads_mat,torch.mv(self.grads_mat.T, self.grads_da))
            self.grads_da -= dot_prod 
            
            # copy gradients back
            overwrite_grad(self.parameters, self.grads_da,self.grad_dims)

        self.opt.step()

        return loss.item()
