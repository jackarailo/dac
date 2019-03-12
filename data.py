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

def create_dataloader(train_data, bsz, device, noise, noise_param, random_seed=None):
    """Return Dataloader object from data"""
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    dataset = Dataset(train_data, device, noise, noise_param)
    dataloader = data.DataLoader(dataset, batch_size=bsz,
                            shuffle=True, num_workers=4)
    return dataloader

class Dataset(data.Dataset):
  """PyTorch Dataset"""
  def __init__(self, data, device, noise, noise_param):
        self.N, self.D = data.shape
        self.data = torch.tensor(data, dtype=torch.float)
        self.targets = torch.tensor(data, dtype=torch.float)
        self.device = device
        self.noise = noise
        self.noise_param = noise_param

  def __len__(self):
        """Return number of data"""
        return self.data.shape[0]

  def __getitem__(self, idx):
        """Return from data with IDX"""
        x = self.data[idx]
        Nx = x.shape[0]
        if self.noise == 'permute':
            p = np.random.choice(np.arange(self.D), replace=False, 
                                size=int(self.D*self.noise_param))
            pidx = np.random.randint(low=0, high=self.N, size=len(p))
            x[p] = self.data[pidx, p]
        elif self.noise == 'gauss':
            x += torch.randn([N,D]) * self.noise_param
        y = self.targets[idx]
        return x, y
