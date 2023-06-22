
import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.ring_buffer import RingBuffer

from datasets import get_dataset


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class TEr(ContinualModel): # TaskEr
    NAME = 'ter'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(TEr, self).__init__(backbone, loss, args, transform)

        # dd = get_dataset(args)
        # self.n_tasks = dd.N_TASKS
        # self.cpt = dd.N_CLASSES_PER_TASK
        self.current_task=0
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if self.current_task > 0: # no replay while learning the first task.
            buf_inputs, buf_labels, _ = self.buffer.get_data_balanced(
                self.args.minibatch_size, self.current_task,  transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()


        task_labels = torch.ones(real_batch_size)*self.current_task
        self.buffer.add_data(examples=not_aug_inputs, 
                             labels=labels[:real_batch_size],
                             task_labels=task_labels)

        return loss.item()
    
    def end_task(self, dataset):
        """
        Updates the task trackers at the end of each task training.
        
        """
        self.current_task +=1
        #self.buffer.task_number = self.current_task
