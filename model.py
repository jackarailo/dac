from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class DACBottle(nn.Module):
    def __init__(self, ninp, nhid, nbot, nhlayers, resnet_trick, random_seed):
        super(DACBottle, self).__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # Bookeeping
        assert nhlayers >= 4, "Min number of layers is 4"
        self.ninp = ninp
        self.nhid = nhid
        self.nbot = nbot
        self.botlayer = nhlayers//2
        self.nhlayers = nhlayers
        self.resnet_trick = resnet_trick

        # Network setup
        fc = [nn.Linear(self.ninp, self.nhid)]
        for i in range(1, self.nhlayers):
            # Intermediate layers nhid x nhid
            if i == self.botlayer:
                fc.append(nn.Linear(self.nhid, self.nbot))
            elif i == self.botlayer + 1:
                fc.append(nn.Linear(self.nbot, self.nhid))
            else:
                fc.append(nn.Linear(self.nhid, self.nhid))
        # Out layer
        fc.append(nn.Linear(self.nhid, self.ninp))
        self.fc = nn.ModuleList(fc)
        for net in self.fc:
            torch.nn.init.kaiming_normal_(net.weight)

    def forward(self, x):
        for i in range(self.nhlayers):
            x = self.fc[i](x)
            a = F.relu(x)
            if a.shape == x.shape and self.resnet_trick:
                x = F.dropout(a, p=self.p) + x
            else:
                x = F.dropout(a, p=self.p)

        out = self.fc[self.nhlayers](x)
        return out

    def generate(self, x):
        for i in range(self.botlayer+1):
            x = self.fc[i](x)
            a = F.relu(x)
            if a.shape == x.shape and self.resnet_trick:
                x = a + x
            else:
                x = a
        return x.cpu().clone().detach().numpy()

class DAC(nn.Module):
    def __init__(self, ninp, nhid, nhlayers, resnet_trick, random_seed):
        super(DAC, self).__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # Bookeeping
        self.ninp = ninp
        self.nhid = nhid
        self.nhlayers = nhlayers
        self.resnet_trick = resnet_trick

        # Network setup
        fc = [nn.Linear(self.ninp, self.nhid)]
        for i in range(1, self.nhlayers):
            # Intermediate layers nhid x nhid
            fc.append(nn.Linear(self.nhid, self.nhid))
        # Out layer
        fc.append(nn.Linear(self.nhid, self.ninp))
        self.fc = nn.ModuleList(fc)
        for net in self.fc:
            torch.nn.init.kaiming_normal_(net.weight)

    def forward(self, x):
        for i in range(self.nhlayers):
            x = self.fc[i](x)
            a = F.relu(x)
            if a.shape == x.shape and self.resnet_trick:
                x = a + x
            else:
                x = a

        out = self.fc[self.nhlayers](x)
        return out

    def generate(self, x):
        out = np.zeros([x.shape[0], self.nhlayers, self.nhid])
        for i in range(self.nhlayers):
            x = self.fc[i](x)
            a = F.relu(x)
            if a.shape == x.shape and self.resnet_trick:
                x = F.dropout(a, p=self.p) + x
            else:
                x = F.dropout(a, p=self.p)
            out[:, i, :] = x.cpu().clone().detach().numpy()
        return out.reshape(N, -1)
