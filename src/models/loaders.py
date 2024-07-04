import numpy as np
import os
import pickle
import torch
from torch.utils.data import TensorDataset

class ClassifierDataset(TensorDataset):
    def __init__(self, datadir, pids, suffix='HeavyAtomsite', resample=True, n_resamples=2000):
        self.datadir = datadir
        self.pids = pids
        self.suffix = suffix
        self.resample = resample
        self.n_resamples = n_resamples

    def __getitem__(self, index):
        sele_pid = self.pids[index]
        propfile = os.path.join(self.datadir, f'{sele_pid}.property.pvar')
        labelfile = os.path.join(self.datadir, f'{sele_pid}.{self.suffix}.labels.pkl')
        with open(propfile, 'rb') as file:
            prop = pickle.load(file)
        with open(labelfile, 'rb') as file:
            labels = pickle.load(file)
        indices = labels[:, 0]
        X = prop[indices.astype(int)]
        Y = labels[:, 1]
        if self.resample:
            zeros = np.sum(Y<0.5)
            ones = np.sum(Y>=0.5)
            probs = [1/(2*ones) if i>=0.5 else 1/(2*zeros) for i in Y]
            resampled_indices = np.random.choice(X.shape[0], \
                size=self.n_resamples, replace=True, p=probs)
            resampled_X = X[resampled_indices]
            resampled_Y = Y[resampled_indices]
            return torch.from_numpy(resampled_X), torch.from_numpy(resampled_Y)
        else:
            return torch.from_numpy(X), torch.from_numpy(Y)
    
    def __len__(self):
        return len(self.pids)