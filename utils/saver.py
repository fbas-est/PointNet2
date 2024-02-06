import os
import shutil
import torch
from collections import OrderedDict
import glob


class Saver(object):

    def __init__(self, args):
        """
        Input:
            args: a set of arguments regarding the overall training algorithm
        """
        self.args = args
        # Define directory for saving trained models

        self.directory = os.path.join('run', self.args.dataset_name, self.args.checkname)
        # Create a directory for storage that wont erase pre-existing directories
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'run_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        self.experiment_dir = os.path.join(self.directory, 'run_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename="last.pth.tar"):
        """
        Saves checkpoint to disk
        Input:
            args: a set of arguments regarding the overall training algorithm
            state: state_dict of the model
            filename: filename of the saved model
        """
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
