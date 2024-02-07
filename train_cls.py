import torch
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
from tqdm import tqdm
import time
from models.pointnet2_cls import PointNet2Classification
from dataloaders.modelnet_dataloader import ModelNetDataloader
from tensorboardX import SummaryWriter
from utils.losses import ClassificationLosses
from utils.saver import Saver
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator


class Trainer(object):

    def __init__(self, args):
        """
        Initialize all the values that are going to be used for the training and validation functions
        Input:
            args: A set of arguments regarding the overall training algorithm
        """

        self.args = args

        # Define the saver for trained models
        self.saver = Saver(args)
        # Define the writer for tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(self.args.log_dir, self.saver.experiment_dir))

        # Define number of available GPUs
        self.args.gpu_ids = [int(s) for s in self.args.gpu_ids.split(',')]
        # Define device for operation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initiate learning rate
        self.learning_rate = self.args.lr

        # Initiate PointNet2 classification model
        self.model = PointNet2Classification(num_class = self.args.number_classes, normal_channel=self.args.use_normals)
        self.model.to(self.device)
        if (len(self.args.gpu_ids) > 1):
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device)

        
        # Resuming checkpoint
        self.start_epoch = 0
        if self.args.resume is not None:
            if not os.path.isfile(self.args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(self.args.resume)
            self.start_epoch = checkpoint['epoch']
            if (len(self.args.gpu_ids) > 1):
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, checkpoint['epoch']))


        # Define Optimizer (2 Options)
        if self.args.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.learning_rate, weight_decay = self.args.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr= self.learning_rate, momentum = self.args.momentum, weight_decay = self.args.weight_decay)

        # Define training and testing datasets
        train_dataset = ModelNetDataloader(os.path.join(self.args.root, self.args.dataset_name), split='train', num_class = self.args.number_classes, 
                                           use_normals=self.args.use_normals, num_points=self.args.num_points, fast_sample=self.args.fast_sample)
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True, num_workers=10)
        test_dataset = ModelNetDataloader(os.path.join(self.args.root, self.args.dataset_name), split='test', num_class = self.args.number_classes, use_normals=self.args.use_normals, num_points=self.args.num_points)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)

        print('The size of train data is %d' % (len(train_dataset)))
        print('The size of test data is %d' % (len(test_dataset)))
        # Define Criterion
        self.criterion = ClassificationLosses(cuda=torch.cuda.is_available()).build_loss(mode=args.loss_type)

        # Define lr scheduler
        self.lr_scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr, self.args.epochs, len(self.train_loader))

        # Define Evaluator
        self.evaluator = Evaluator(self.args.number_classes)

    def train_one_epoch(self, epoch):
        """
        Input:
            epoch: An int value that defines the current epoch of the training procedure
        """
        print("===TRAINING===")
        # Define a variable for holding the overall loss accross the dataset
        train_loss = 0.0
        # Enable training operation for the Neural Network
        self.model.train()
        # Use tqdm for visualizing the progress bar of the training
        tbar = tqdm(self.train_loader)
        for i, batch in enumerate(tbar):
            
            # Get the values from the batch of the train_loader
            points, labels, cls_names = batch
            
            points = points.to(self.device)
            labels = labels.to(self.device)
            
            # Adjust lr value
            self.learning_rate = self.lr_scheduler(self.optimizer, i, epoch)
            # Clear gradients before computing anew
            self.optimizer.zero_grad()
            # Make a forward propagation (prediction) of the model
  
            pred = self.model(points)

            # Calculate the loss for this batch
            loss = self.criterion(pred[0], labels)

            # Backpropagation for gradients calculation
            loss.backward()
            # Update model parameters using the computed gradients
            self.optimizer.step()

            # Add the current loss value to the overall loss for the training dataset
            train_loss += loss.item()

            # Visualize loss in the progress bar
            tbar.set_description('||Train loss: %.5f || Learning Rate: %.7f || Epoch/Total_Epochs: %d/%d ||' % (loss.item(), self.learning_rate, epoch, self.args.epochs))
            # Update loss value in the tensorboard schema
            self.writer.add_scalar('train/batch_train_loss', loss.item(), i + len(self.train_loader) * epoch)


        # Calculate the average of the loss accross the dataset
        avg_loss_train = train_loss/len(self.train_loader)
        self.writer.add_scalar('train/train_loss', avg_loss_train, epoch)
        print('[Epoch: %d, Loss: %.3f]' % (epoch, avg_loss_train))

        if not self.args.no_val:
            if self.args.save_checkpoint:
                if epoch!=0 and epoch % self.args.save_epochs == 0:
                    # Save model if Parallel training is enabled
                    if (len(self.args.gpu_ids) > 1):
                        self.saver.save_checkpoint({'epoch': epoch + 1, 
                                                    'learning_rate': self.learning_rate,
                                                    'state_dict': self.model.module.state_dict(), 
                                                    'optimizer': self.optimizer.state_dict()},
                                                    filename=f"checkpoint_parallel_{epoch}.pth.tar")
                    # Save model trained on a single GPU
                    else:
                        self.saver.save_checkpoint({'epoch': epoch + 1, 
                                            'learning_rate': self.learning_rate,
                                            'state_dict': self.model.state_dict(), 
                                            'optimizer': self.optimizer.state_dict()},
                                            filename=f"checkpoint_{epoch}.pth.tar")
                    
            if (len(self.args.gpu_ids) > 1):
                self.saver.save_checkpoint({'epoch': epoch + 1, 
                                            'learning_rate': self.learning_rate,
                                            'state_dict': self.model.module.state_dict(), 
                                            'optimizer': self.optimizer.state_dict()},
                                            filename="last_parallel.pth.tar")
            else:
                self.saver.save_checkpoint({'epoch': epoch + 1, 
                                    'learning_rate': self.learning_rate,
                                    'state_dict': self.model.state_dict(), 
                                    'optimizer': self.optimizer.state_dict()})
        
    def validation(self, epoch):
        """
        Input:
            epoch: An int value that defines the current epoch of the training procedure
        """
        print("===VALIDATION===")
        # Define a variable for holding the overall loss accross the dataset
        test_loss = 0.0
        # Enable evaluation operation for the Neural Network
        self.model.eval()
        # Reset the evaluator's confusion matrix
        self.evaluator.reset()
        # Use tqdm for visualizing the progress bar of the testing
        tbar = tqdm(self.test_loader)
        
        for i, batch in enumerate(tbar):

            # Get the values from the batch of the train_loader
            points, labels, cls_names = batch
            # Pass them to the processing device
            points = points.to(self.device)
            labels = labels.to(self.device)

            # Make a forward propagation (prediction) of the model with gradients disabled
            with torch.no_grad():
                pred, pnts = self.model(points)
        

            # Calculate the loss for this batch
            loss = self.criterion(pred, labels)
            # Add the current loss value to the overall loss for the validation dataset
            test_loss += loss.item()

            pred_np = pred.data.cpu().numpy()
            labels_np = labels.cpu().numpy()
            # Calculate confusion matrix
            self.evaluator.add_batch(np.argmax(pred_np), labels_np[0])

            # Visualize loss in the progress bar
            tbar.set_description('||Test loss: %.5f ||' % (loss.item()))
        
        # Calculate the Accuracy metric
        Acc = self.evaluator.cal_acc()
        # Calculate the F1 score metric
        F1_score = self.evaluator.cal_f1_score()    
        # Calculate the avg loss in the testing dataset    
        avg_loss_test = test_loss/len(self.test_loader)

        # Update tennsorboard
        self.writer.add_scalar('val/test_loss', avg_loss_test, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/F1_score', F1_score, epoch)
        
        print('[Epoch: %d, Loss: %.3f]' % (epoch, avg_loss_test))
        print("Acc:{}, F1_score:{}".format(Acc, F1_score))
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data', help='root')
    parser.add_argument('--dataset_name', type=str, default='modelnet40_normal_resampled', help='dataset name')
    parser.add_argument('--log_dir', type=str, default='log_dir', help='log dir for Summary')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--number_classes', type=int, default=40, help='Number of training classes')
    parser.add_argument('--num_points', type=int, default=None, help='Point Number')
    parser.add_argument('--fast_sample', action='store_true', default=False, help='Point Number')


    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of batches")
    parser.add_argument("--optim", type=str, default='Adam', help="optimizer")
    parser.add_argument("--loss_type", type=str, default='ce', help="Define which loss function is going to be used")


    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate [default: 1e-2]")
    parser.add_argument('--lr_scheduler', type=str, default='poly', choices=['poly', 'step', 'cos'], help="Type of scheduler for adjusting learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization coeff [default: 0.0]")
    parser.add_argument("--momentum", type=float, default=0.92, help="Momentum  to escape local minima")

    parser.add_argument('--no_val', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--save_checkpoint', action='store_true', default=True, help='Enable saving of checkpoints, if not enabled only the last model will be saved')
    parser.add_argument("--save_epochs", type=int, default=5, help="Number of epochs in which the model will be saved")
    parser.add_argument('--checkname', type=str, default='pointnet2_cls', help='set the checkpoint name')
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to start from")
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--gpu_ids', type=str, default="0", help="Seperate different GPU ids with comma")
    args = parser.parse_args()

    trainer = Trainer(args)
    print('Total Epoches:', trainer.args.epochs)
    
    for epoch in range(trainer.start_epoch, trainer.args.epochs):
        trainer.train_one_epoch(epoch)
        if not trainer.args.no_val and epoch % 1 == 0:
            trainer.validation(epoch)
        
    trainer.writer.close()
  
