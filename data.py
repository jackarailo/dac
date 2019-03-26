import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from torch.utils import data

def load(f):
    """Return pandas dataframe from csv F"""
    return pd.read_csv(f)

def normalize(df, norm_type='std'):
    """Return normalized ndarray from DF"""
    data = df.values
    if norm_type == 'std':
        return prpprocessing.StandardScaler().fit_transform(data)
    elif norm_type == 'rankGauss':
        sc = preprocessing.QuantileTransformer(output_distribution='normal')
        return sc.fit_transform(data)
    elif norm_type == 'minmax':
        return preprocessing.MinMaxScaler().fit_transform(data)

def create_dataloader(targets, bsz, device, noise, noise_param, random_seed=None):
    """Return Dataloader object from data"""
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    train_data = targets.copy()
    N, D = train_data.shape
    if noise == 'permute':
        Da = np.arange(D)
        for i in range(N):
            p = np.random.choice(Da, replace=False, 
                                size=int(D*noise_param))
            pidx = np.random.randint(low=0, high=N, size=int(D*noise_param))
            for j in range(len(pidx)):
                train_data[i, p[j]] = targets[pidx[j], p[j]]
    elif noise == 'gauss':
        train_data += torch.randn([N,D]) * noise_param
    dataset = Dataset(train_data, targets, device)
    dataloader = data.DataLoader(dataset, batch_size=bsz,
                            shuffle=True, num_workers=4)
    return dataloader

class Dataset(data.Dataset):
  """PyTorch Dataset"""
  def __init__(self, train_data, targets, device):
        self.N, self.D = train_data.shape
        self.data = torch.tensor(train_data, dtype=torch.float)
        self.targets = torch.tensor(targets, dtype=torch.float)
        self.device = device

  def __len__(self):
        """Return number of data"""
        return self.data.shape[0]

  def __getitem__(self, idx):
        """Return from data with IDX"""
        x = self.data[idx]
        y = self.targets[idx]
        return x, y
