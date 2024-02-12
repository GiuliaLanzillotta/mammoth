"""Takes a job run command and prints out a sequence of commands with multiple seeds 
dividing the GPUs equally between them

example commands: 

python run_all.py python utils/main.py --model er --dataset seq-tinyimg --buffer_size 5120 --load_best_args --notes ertinyimg 

"""
import os
import sys
from copy import copy
import subprocess


SEEDS = [11,13,21,33,55]
LR = [0.0001, 0.001, 0.01, 0.05, 0.1]
PARALLEL_ORDER = 4
GPUIDS = [0, 1, 2, 3]
SAVE_CHKPTS = False

def crange(start, end, modulo):
    # implementing circular range
    if start > end:
        while start < modulo:
            yield start
            start += 1
        start = 0

    while start < end:
        yield start
        start += 1


all_commands=[]
gpu_count=0
job_count=0

for lr in LR: 
    for seed in SEEDS:
        new_argv = copy(sys.argv)
        new_argv.append(f'--seed {seed} ')
        new_argv.append(f'--lr {lr}')
        new_argv.append(f'--gpus_id {GPUIDS[job_count]}')
        if seed in [11,13] and SAVE_CHKPTS: 
                new_argv.append('--savecheckpoints')

        gpu_idx = job_count % len(GPUIDS)
        new_argv.append(f'--gpus_id {GPUIDS[gpu_idx]}')
        
        job_count+=1
        
        all_commands.append(" ".join(new_argv[1:]))
        if job_count==PARALLEL_ORDER:
            subprocess.run(["utils/run_commands.sh"]+all_commands)
            all_commands=[]
            job_count=0 

if len(all_commands)>0:
     # sending the last jobs in the queue
     subprocess.run(["utils/run_commands.sh"]+all_commands)