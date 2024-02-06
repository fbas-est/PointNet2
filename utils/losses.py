import torch
import torch.nn as nn


class ClassificationLosses(object):

    def __init__(self, weight=None, size_average=True, batch_average=True, cuda=False):
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """
        Input:
            mode: ['ce', 'nll'] Select loss function
        """
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'nll':
            return self.nll_loss
        
    def CrossEntropyLoss(self, logit, target):
        n, _= logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss
    
    def nll_loss(self, logit, target):
        n, _ = logit.size()

        criterion = nn.NLLLoss(weight = self.weight, size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())
        
        if self.batch_average:
            loss /= n

        return loss