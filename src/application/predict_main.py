import os, sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
filename = os.path.abspath(__file__)
parentdir = os.path.dirname(os.path.dirname(filename))
sys.path.append(parentdir)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="""
    Apply trained model.
""")
parser.add_argument('--configfile', type=str,\
    help='Configuration torch file')
parser.add_argument('--paramfile', type=str,\
    help='Parameter file')
parser.add_argument('--batchsize', type=int, default=10000, \
    help='batch size used in dataloader')
parser.add_argument('--datafile', type=str, \
    help='path to feature file')
parser.add_argument('--outfile', type=str, \
    help='path to store results')
parser.add_argument('--use_gpu', action='store_true', default=False, \
    help='Use gpu when gpu is available')
args = parser.parse_args()

if args.use_gpu and torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

CONFIG = torch.load(args.configfile, map_location=torch.device(DEVICE))

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


def create_predloader_fromfile(propfile, batch_size=1024):
    with open(propfile, 'rb') as file:
        prop_arr = pickle.load(file)
    print('Shape of FEATURE vectors to predict:', prop_arr.shape)
    dataloader = DataLoader(torch.from_numpy(prop_arr), batch_size=batch_size, shuffle=False)
    return dataloader

def predict(dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x in dataloader:
            logits = model(x.float().to(DEVICE))
            p = torch.softmax(logits, dim=1)
            predictions.extend(p.cpu().numpy().tolist())
    return np.array(predictions)


dataloader = create_predloader_fromfile(args.datafile, args.batchsize)
pred_probs = predict(dataloader)
with open(args.outfile, 'wb') as file:
    pickle.dump(pred_probs, file)
print(f"Finish predicting {args.datafile}")
            