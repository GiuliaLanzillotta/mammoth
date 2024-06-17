# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
import numpy as np

WANDBKEY = "c7314b16e17009f66f3043c0b7968e1142054123"

def get_device(gpus_id:list) -> torch.device:
    """
    Returns the GPU device corresponding to the first list index if available else CPU.
    """

    if torch.cuda.is_available() and len(gpus_id)>0:
        return torch.device(gpus_id[0])
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    except:
        pass
    return torch.device("cpu")


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './logs/'

def base_path_dataset() -> str:
    """
    Returns the base path where to store data.
    """
    return '/media/hofmann-scratch/glanzillo/mammoth/data/'


def base_path_checkpoints() -> str:
    """
    Returns the base path where to find checkpoints.
    """
    return '/media/hofmann-scratch/glanzillo/mammoth/logs/chkpts/'

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        print('Could not set cuda seed.')
        pass
