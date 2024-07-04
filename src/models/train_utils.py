import os
import re
import glob
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_all_data(datafolder, pid_pool):
    DCT = {}
    for pdbid in pid_pool:
        propfile = os.path.join(datafolder, f'{pdbid}.property.pvar')
        labelfile = os.path.join(datafolder, f'{pdbid}.voxel.labels.pkl')
        with open(propfile, 'rb') as file:
            prop_arr = pickle.load(file)
        with open(labelfile, 'rb') as file:
            label_dct = pickle.load(file)
        X = prop_arr
        n = X.shape[0]
        Y = [[] for _ in range(n)]
        for key, value in label_dct.items():
            for idx in value:
                Y[idx].append(key)
        DCT[pdbid] = (X, Y)
    return DCT

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_foldids(test_fold, foldids):
    """
    Return
    ------
    train_foldid, val_foldid, test_foldid : list
    """
    mod = len(set(foldids))
    test_foldid = [test_fold%mod]
    val_foldid = sorted([(test_fold+1)%mod, (test_fold+2)%mod])
    train_foldid = set(foldids) - set(test_foldid+val_foldid)
    train_foldid = sorted(train_foldid)
    return train_foldid, val_foldid, test_foldid


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)


def calc_struct_probs(foldids, df, pid_pool=None):
    """
    Calculate sample probability of each structure.
    foldids :  a list of foldid
        df['Structure_Clan_ID']
    df : organised structure clan
    pid_pool : a list of pdb ids
        This list is used in case structure files do not exist
    Return
    ------
    todo_pids, todo_probs
    """
    mask = df['Structure_Clan_ID'].apply(lambda x: x in foldids)
    pids = df[mask]['PDBIDList'].tolist()
    if pid_pool is not None:
        pids = [sorted(set(lst).intersection(pid_pool)) for lst in pids]
    clan_sizes = [len(lst) for lst in pids]
    todo_pids = []
    todo_probs = []
    clans = 0
    for lst, size in zip(pids, clan_sizes):
        if size > 0:
            todo_pids.extend(lst)
            todo_probs.extend([1/size]*size)
            clans += 1
    if clans == 0:
        return [], []
    else:
        todo_probs = [p/clans for p in todo_probs]
        return todo_pids, todo_probs


def create_predloader_fromfile(pid, folder, batch_size=1024):
    propfile = os.path.join(folder, f'{pid}.property.pvar')
    with open(propfile, 'rb') as file:
        prop_arr = pickle.load(file)
    dataloader = DataLoader(torch.from_numpy(prop_arr), batch_size=batch_size, shuffle=False)
    return pid, dataloader
