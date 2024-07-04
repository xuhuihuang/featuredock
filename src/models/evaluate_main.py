import os, sys
import argparse
import pickle
import itertools
import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
filename = os.path.abspath(__file__)
parentdir = os.path.dirname(os.path.dirname(filename))
sys.path.append(parentdir)
from models.loaders import ClassifierDataset
from models.train_utils import init_seed
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="""
    Evaluate model with validation set.
""")
parser.add_argument('--seed', type=int, help='random seed to help reproduce results')
parser.add_argument('--configfile', type=str,\
    help='Configuration torch file')
parser.add_argument('--paramfile', type=str,\
    help='Parameter file')
parser.add_argument('--datafolder', type=str, default=None, \
    help='datafolder')
parser.add_argument('--tqdm_disable', default=False, action='store_true', \
    help='Disable to print tqdm progress')
parser.add_argument('--use_gpu', action='store_true', default=False, \
    help='Use gpu when gpu is available')
args = parser.parse_args()

init_seed(args.seed)

## load configuration
if args.use_gpu and torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
CONFIG = torch.load(args.configfile, map_location=torch.device(DEVICE))


clanfile = CONFIG['graphclan']
with open(clanfile, 'rb') as file:
    df = pickle.load(file)
val_foldids = CONFIG['val_foldids']
print('val_foldids', val_foldids)


## restore model
model = CONFIG['model']
model = model.float()
model.to(DEVICE)
print('Load params from checkpoint file.')
checkpoint = torch.load(args.paramfile, map_location=torch.device(DEVICE))
try:
    model.load_state_dict(checkpoint['model_state_dict'])
except:
    model.load_state_dict(checkpoint)
loss_function = nn.CrossEntropyLoss()

def evaluate(foldids, CONFIG, cls_balance=False):
    suffix = CONFIG['task']
    pid_pool = CONFIG['pid_pool']
    datafolder = args.datafolder if args.datafolder else CONFIG['datafolder']
    mask = df['Structure_Clan_ID'].apply(lambda x: x in foldids)
    test_pids = df[mask]['PDBIDList'].tolist()
    test_pids = [set(lst).intersection(pid_pool) for lst in test_pids]
    test_pids = sorted(list(itertools.chain(*test_pids)))

    if cls_balance:
        test_dataset = ClassifierDataset(datafolder, test_pids, suffix=suffix, resample=True)
        testloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    else:
        test_dataset = ClassifierDataset(datafolder, test_pids, suffix=suffix, resample=False)
        testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    ## evaluate
    model.eval()
    losses = []
    precs = []
    recalls = []
    fscores = []
    accs = []
    mccs = []
    predictions = []
    truths = []
    aucs = []
    counts = []
    with torch.no_grad():
        for x, y in tqdm.tqdm(testloader, disable=args.tqdm_disable):
            shapes = x.shape
            x = x.reshape((shapes[0]*shapes[1], *shapes[2:]))
            y = y.reshape((-1, 1))
            y = torch.cat([torch.ones_like(y)-y, y], dim=1)
            prediction = model(x.float().to(DEVICE))
            p = torch.softmax(prediction, dim=1)
            y_pred = torch.argmax(p, dim=1)
            y_true = torch.argmax(y, dim=1)
            loss = loss_function(torch.Tensor(prediction), \
                                torch.argmax(y.float().to(DEVICE), dim=1))
            y_pred = y_pred.cpu().numpy().tolist()
            y_true = y_true.cpu().numpy().tolist()
            f_score = score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            # metrics
            counts.append(len(y_pred))
            losses.append(float(loss))
            precs.append(f_score[0][1])
            recalls.append(f_score[1][1])
            fscores.append(f_score[2][1])
            accs.append(acc)
            mccs.append(mcc)
            aucs.append(auc)
            predictions.extend(y_pred)
            truths.extend(y_true)
    auc = roc_auc_score(truths, predictions)
    f_score = score(truths, predictions)
    acc = accuracy_score(truths, predictions)
    mcc = matthews_corrcoef(truths, predictions)
    total_loss = [count*loss for count, loss in zip(counts, losses)]
    avg_loss = np.sum(total_loss) / np.sum(counts)
    return np.mean(aucs), np.mean(precs), np.mean(recalls), \
        np.mean(fscores), np.mean(accs), np.mean(mccs), np.mean(losses), \
        auc, f_score[0][1], f_score[1][1], f_score[2][1], acc, mcc, avg_loss

stats = evaluate(val_foldids, CONFIG, cls_balance=False)
print('Validation Set Imbalanced (Per Structure)')
print(f'Loss={stats[6]}\nAUC={stats[0]}\nPrecision={stats[1]}\nRecall={stats[2]}\nFScore={stats[3]}\nAccuracy={stats[4]}\nMCC={stats[5]}\n')
print('Validation Set Imbalanced (Per Point)')
print(f'Loss={stats[13]}\nAUC={stats[7]}\nPrecision={stats[8]}\nRecall={stats[9]}\nFScore={stats[10]}\nAccuracy={stats[11]}\nMCC={stats[12]}\n')
print('='*20)

stats = evaluate(val_foldids, CONFIG, cls_balance=True)
print('Validation Set Balanced (Per Structure)')
print(f'Loss={stats[6]}\nAUC={stats[0]}\nPrecision={stats[1]}\nRecall={stats[2]}\nFScore={stats[3]}\nAccuracy={stats[4]}\nMCC={stats[5]}\n')
print('Validation Set Balanced (Per Point)')
print(f'Loss={stats[13]}\nAUC={stats[7]}\nPrecision={stats[8]}\nRecall={stats[9]}\nFScore={stats[10]}\nAccuracy={stats[11]}\nMCC={stats[12]}\n')
print('='*20)
