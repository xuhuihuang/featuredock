import os
import sys
import argparse
import datetime
import time
import json
import glob
import itertools
import pickle
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, matthews_corrcoef

filename = os.path.abspath(__file__)
parentdir = os.path.dirname(os.path.dirname(filename))
sys.path.append(parentdir)
from models.train_utils import init_seed, count_parameters, calc_struct_probs
from models.loaders import ClassifierDataset
from models.earlystop import EarlyStopping
from models.parse_config import parser, load_config
torch.cuda.empty_cache()

##################
#  LOAD CONFIG   #
##################
args = parser()
CONFIG = load_config(args)


#############
# OUTFOLDER #
#############
if not os.path.exists(CONFIG['outfolder']):
    os.makedirs(CONFIG['outfolder'], exist_ok=True)
    print(f'Output folder created: {CONFIG["outfolder"]}')

############
#  DEVICE  #
############
print('Determine device')
if  CONFIG['has_gpu'] and CONFIG['use_gpu']:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
print(f'Using {DEVICE}.')

###################
#  SPLIT DATASET  #
###################
FOLDER = CONFIG['datafolder']
print(f'Data folder: {FOLDER}.')
files = list(glob.glob(os.path.join(FOLDER, '*.property.pvar')))
PID_POOL = [os.path.basename(fname).split('.')[0] for fname in files]
print('Total number of structures in the pool:', len(PID_POOL))

print('Split protein structure clans to Train/Val')
clanfile = CONFIG['graphclan']
with open(clanfile, 'rb') as file:
    DF = pickle.load(file)
FOLDIDS = DF['Structure_Clan_ID'].unique().tolist()
test_foldids = [-1] if -1 in FOLDIDS else []
ratio = 0.1
rest = sorted(set(FOLDIDS)-set(test_foldids))
val_foldids = np.random.choice(rest, int(np.ceil(ratio*len(rest))), replace=False)
val_foldids = sorted(val_foldids)
train_foldids = set(FOLDIDS)-set(test_foldids+val_foldids)
train_foldids = sorted(train_foldids)
CONFIG.update({
    'pid_pool': PID_POOL,
    'train_foldids': train_foldids,
    'val_foldids': val_foldids,
    'test_foldids': test_foldids
})

##############
#   MODEL    #
##############
loss_function = nn.CrossEntropyLoss()
model = CONFIG['model']
model = model.float()
model.to(DEVICE)
optimizer = CONFIG['optimizer']
scheduler = CONFIG['scheduler']

#######################
#  LOAD CKPT IF ANY   #
#######################
if CONFIG['checkpoint'] is not None:
    print('Load params from checkpoint file.')
    checkpoint = torch.load(CONFIG['checkpoint'], map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load(checkpoint['scheduler_state_dict'])
param_num = count_parameters(model)
print('Number of tunable parameters:', param_num)


################
# WRITE CONFIG #
################
config_file = os.path.join(CONFIG['outfolder'], f'{CONFIG["modelname"]}_config.torch')
torch.save(CONFIG, config_file)
print('Configuration:', CONFIG)

###########################
#  DEFINE TRAIN FUNCTIONS #
###########################
def train_one_step(dataloader):
    model.train()
    scores = []
    losses = []
    mccs = []
    for x, y in dataloader:
        shapes = x.shape
        x = x.reshape((shapes[0]*shapes[1], *shapes[2:]))
        y = y.reshape((-1, 1))
        y = torch.cat([torch.ones_like(y)-y, y], dim=1)
        output = model(x.float().to(DEVICE))
        optimizer.zero_grad()
        loss = loss_function(output, torch.argmax(y.float().to(DEVICE), dim=1))
        loss.backward()
        optimizer.step()
        p = torch.softmax(output, dim=1)
        y_pred = torch.argmax(p, dim=1)
        y_true = torch.argmax(y, dim=1)
        f_score = score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        acc_score = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        mcc = matthews_corrcoef(y_true.cpu().numpy(), y_pred.cpu().numpy())
        losses.append(float(loss))
        scores.append((f_score, acc_score))
        mccs.append(mcc)
    avg_acc = np.mean([item[1] for item in scores])
    avg_loss = np.mean(losses)
    stats = [item[0] for item in scores]
    avg_prec = np.mean(np.vstack([i[0] for i in stats]), axis=0)
    avg_recall = np.mean(np.vstack([i[1] for i in stats]), axis=0)
    avg_f1_score = np.mean(np.vstack([i[2] for i in stats]), axis=0)
    avg_mcc = np.mean(mccs)
    return avg_loss, avg_acc, avg_prec, avg_recall, avg_f1_score, avg_mcc

def val_one_step(dataloader):
    model.eval()
    losses = []
    predictions = []
    truth = []
    with torch.no_grad():
        for x, y in dataloader:
            shapes = x.shape
            x = x.reshape((shapes[0]*shapes[1], *shapes[2:]))
            y = y.reshape((-1, 1))
            y = torch.cat([torch.ones_like(y)-y, y], dim=1)
            prediction = model(x.float().to(DEVICE))
            p = torch.softmax(prediction, dim=1)
            y_pred = torch.argmax(p, dim=1)
            y_true = torch.argmax(y, dim=1)
            loss = loss_function(torch.Tensor(prediction), torch.argmax(y.float().to(DEVICE), dim=1))
            losses.append(float(loss))
            predictions.extend(y_pred.cpu().numpy().tolist())
            truth.extend(y_true.cpu().numpy().tolist())
    f_score = score(truth, predictions)
    acc = accuracy_score(truth, predictions)
    mcc = matthews_corrcoef(truth, predictions)
    return np.mean(losses), acc, f_score[0], f_score[1], f_score[2], mcc


def run_one_fold(modelname, splits, df, folder, suffix, pid_pool, CONFIG, verbose=True):
    ## load hyperparameters
    n_structs, n_resamples = CONFIG['n_structs'], CONFIG['n_resamples']
    outfolder = CONFIG['outfolder']
    scheduler_name = CONFIG['scheduler_name']
    train_foldid, val_foldid, test_foldid = splits
    param_file = os.path.join(outfolder, f'{modelname}_final_params.torch')
    if os.path.exists(param_file):
        raise FileExistsError(f"""Trained model already exists: {param_file}.
            Please delete manually to avoid overwriting.""")
    ## save config for further validation
    ## sample datasets
    train_pids, train_probs = calc_struct_probs(train_foldid, df, pid_pool)
    train_clusters = df['Structure_Clan_ID'].apply(lambda x: x in train_foldid).sum()
    train_sampler = WeightedRandomSampler(train_probs, int(train_clusters), replacement=True)
    val_pids, val_probs = calc_struct_probs(val_foldid, df, pid_pool)
    val_clusters = df['Structure_Clan_ID'].apply(lambda x: x in val_foldid).sum()
    val_sampler = WeightedRandomSampler(val_probs, int(val_clusters), replacement=True)
    if len(test_foldid) == 0:
        test_pids = []
    else:
        mask = df['Structure_Clan_ID'].apply(lambda x: x in test_foldid)
        test_pids = df[mask]['PDBIDList'].tolist()
        test_pids = [set(lst).intersection(pid_pool) for lst in test_pids]
        test_pids = sorted(list(itertools.chain(*test_pids)))
    history = []
    if verbose:
        print(f'Number of training structures: {len(train_pids)}')
        # print(f'Train structures: {dict(zip(train_pids, train_probs))}')
        print(f'Number of validation structures: {len(val_pids)}')
        # print(f'Validation structures: {dict(zip(val_pids, val_probs))}')
        print(f'Number of test structures: {len(test_pids)}')
        print(f'Test structures: {test_pids}')
    
    train_dataset = ClassifierDataset(folder, train_pids, suffix=suffix, resample=True, n_resamples=n_resamples)
    trainloader = DataLoader(train_dataset, batch_size=n_structs, sampler=train_sampler)
    val_dataset = ClassifierDataset(folder, val_pids, suffix=suffix, resample=True, n_resamples=n_resamples)
    valloader = DataLoader(val_dataset, batch_size=n_structs, sampler=val_sampler)
    if len(test_pids) == 0:
        testloader = None
    else:
        test_dataset = ClassifierDataset(folder, test_pids, suffix=suffix, resample=False)
        testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    if CONFIG['earlystop']:
        print('Using early stopping.')
        patience = CONFIG['patience']
        best_checkpoint = os.path.join(outfolder, f'{modelname}_best_checkpoint_params.torch')
        early_stopping = EarlyStopping(patience=patience, path=best_checkpoint)

    for step in tqdm.tqdm(range(CONFIG['steps']), disable=args.tqdm_disable):
        stats = train_one_step(trainloader)
        val_stats = val_one_step(valloader)
        if testloader is not None:
            test_stats = val_one_step(testloader)
        else:
            test_stats = None
        history.append({
            'stats': stats,
            'val_stats': val_stats,
            'test_stats': test_stats,
            'lr': optimizer.state_dict()['param_groups'][0]['lr'],
        })
        if verbose:
            print('='*20+str(step)+'='*20)
            print('Lerning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            print('TRAIN:', stats)
            print('VALIDATION:', val_stats)
            print('TEST:', test_stats)
        if step % CONFIG['save_every'] == 0:
            checkpoint = os.path.join(outfolder, f'{modelname}_checkpoint_step{step}_params.torch')
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheuler_state_dict': scheduler.state_dict()},
                checkpoint)
        if scheduler is not None:
            if scheduler_name == 'plateau':
                scheduler.step(val_stats[0])
            else:
                scheduler.step()
        if CONFIG['earlystop']:
            early_stopping(val_stats[0], model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    log_file = os.path.join(outfolder, f'{modelname}_logs.pkl')
    with open(log_file, 'wb') as file:
        pickle.dump(history, file)
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheuler_state_dict': scheduler.state_dict()},
        param_file)

##########
#  TRAIN #
##########
run_one_fold(CONFIG['modelname'], [train_foldids, val_foldids, test_foldids], \
    DF, FOLDER, CONFIG['task'], PID_POOL, CONFIG, verbose=args.verbose)