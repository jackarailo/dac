import time
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Trainer():
    def __init__(self, model, lr, lr_decay, reg):
        super(Trainer, self).__init__()
        self.reg = reg
        self.best_val_loss = np.inf
        self.lr_decay = lr_decay
        self.model = model
        params_dict = dict(model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if key[-4:] == 'bias' or key[:2] == 'bn':
                params += [{'params':value,'weight_decay':0.0}]
            else:
                params += [{'params':value,'weight_decay':self.reg}]

        self.params = params
        self.optim = torch.optim.Adam(self.params, 
                                      lr=lr, weight_decay=self.reg)

        if self.lr_decay:
            lambd = lambda epoch: 0.995 ** epoch
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, 
                                                               lr_lambda=lambd)
        self.criterion = nn.MSELoss()
        self.model.train()

    def step(self, x, y):
        self.optim.zero_grad()
        x = self.model(x)
        loss = self.criterion(x, y)
        loss.backward()
        self.optim.step()
        if self.lr_decay:
            self.scheduler.step()
        return loss.item()
