import os
import sys
import argparse
import datetime
import torch
import torch.optim as optim
filename = os.path.abspath(__file__)
parentdir = os.path.dirname(os.path.dirname(filename))
sys.path.append(parentdir)
from models.train_utils import init_seed

def parser():
    parser = argparse.ArgumentParser(description="""
        Training.
    """)
    parser.add_argument('--modeltype', type=str, default='fnn', \
        help='Choose from cnn/fnn/transformer')
    parser.add_argument('--seed', type=int, help='random seed to help reproduce results')
    parser.add_argument('--task', type=str, default='HeavyAtomsite', \
        help='Choose from HeavyAtomsite, HBAcceptor, HBDonor, ...')
    parser.add_argument('--modelname', type=str, default=None, \
        help='Sepecify a model name, the default name is <type>_<time>')
    parser.add_argument('--n_blocks', type=int, default=10, \
        help='Number of ResNet blocks in cnn model')
    parser.add_argument('--h_dims', type=int, nargs='+', default=[], \
        help='Dimension of hidden layers in fnn')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'AdamW'], \
        default='AdamW', help='optimizer')
    parser.add_argument('--scheduler', type=str, choices=['exponential', 'plateau', 'cosine'], \
        default=None, help='scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, \
        help='learning rate')
    parser.add_argument('--steps', type=int, default=30, \
        help='Number of steps to train')
    parser.add_argument('--n_structs', type=int, default=5, \
        help='Number of structures to read (batch_size in dataloader)')
    parser.add_argument('--n_resamples', type=int, default=1000, \
        help='Number of datapoints after balancing for each structure')
    parser.add_argument('--weight_decay', type=float, default=0, \
        help='weight decay used in optimizer')
    parser.add_argument('--save_every', type=int, default=10, \
        help='Save every n steps')
    parser.add_argument('--earlystop', default=False, action='store_true', \
        help='Apply early stop strategy when flagged')
    parser.add_argument('--patience', type=int, default=10, \
        help='number of steps used in early stopping')
    parser.add_argument('--checkpoint', type=str, default=None, \
        help='Load from checkpoint file when provided')
    parser.add_argument('--graphclan', type=str,\
        help='Grapgclan to use')
    parser.add_argument('--datafolder', type=str, \
        help='Folder path to curated dataset: property files and labels')
    parser.add_argument('--outfolder', type=str, \
        help='Folder path to store output files: training logs, model params, ...')
    parser.add_argument('--verbose', default=False, action='store_true', \
        help='Print more details when flagged')
    parser.add_argument('--tqdm_disable', default=False, action='store_true', \
        help='Disable to print tqdm progress')
    parser.add_argument('--use_gpu', default=False, action='store_true', \
        help='Enable gpu')
    args = parser.parse_args()
    return args


def load_config(args):
    CONFIG = {'n_class':2, 'has_gpu': torch.cuda.is_available(), 'use_gpu': args.use_gpu, 'seed': args.seed}
    ################
    #  INIT SEED   #
    ################
    init_seed(CONFIG['seed'])
    print(f"Set random seed as {CONFIG['seed']}")
    ################
    #  BUILD MDL   #
    ################
    if args.modeltype == 'cnn':
        from models.customise_models import FeatIntResNet
        CONFIG['n_blocks'] = args.n_blocks
        CONFIG['lr'] = args.lr
        model = FeatIntResNet(feature_per_shell=80, num_shells=6, n_blocks=CONFIG['n_blocks'], \
                n_mix=2, n_class=CONFIG['n_class'], activation='relu', noise=None)
    elif args.modeltype == 'fnn':
        from models.customise_models import get_fnn_model
        CONFIG['h_dims'] = args.h_dims
        CONFIG['lr'] = args.lr
        model = get_fnn_model(480, CONFIG['h_dims'], CONFIG['n_class'], dropout=0.5)
    elif args.modeltype == 'transformer':
        from models.transformer_models import BertSentClassifier
        CONFIG['n_blocks'] = args.n_blocks
        CONFIG['lr'] = args.lr
        CONFIG['hidden_size'] = 64
        CONFIG['intermediate_size'] = 64
        CONFIG['num_attention_heads'] = 2
        model = BertSentClassifier(n_class=CONFIG['n_class'], num_shells=6, feature_per_shell=80, \
                                hidden_size=CONFIG['hidden_size'], \
                                intermediate_size=CONFIG['intermediate_size'], 
                                num_hidden_layers=CONFIG['n_blocks'], \
                                num_attention_heads=CONFIG['num_attention_heads'], \
                                max_position_embeddings=100, layer_norm_eps=1e-12, \
                                hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, option='finetune')
    else:
        raise NotImplementedError('Not implemented yet. Please select between valid models: cnn/fnn')

    if args.optimizer == 'SGD':
        optimizer = getattr(optim, 'SGD')(model.parameters(), lr=CONFIG['lr'], \
                        momentum=0.95, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'AdamW':
        optimizer = getattr(optim, 'AdamW')(model.parameters(), lr=CONFIG['lr'], \
                        weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not supported yet")

    if args.scheduler=='plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                    patience=1, factor=0.5, mode='min',
                                    verbose=True,
                                    threshold_mode='rel',
                                    threshold=1e-3,
                                    min_lr=1e-6, cooldown=1)
    elif args.scheduler=='cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, \
                    T_0=20, T_mult=1, eta_min = 1e-6)
    elif args.scheduler=='exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9, verbose=True)
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError(f'Scheduler {args.scheduler} not implemented.')

    CONFIG.update({
        'model': model,
        'optimizer': optimizer,
        'scheduler_name': args.scheduler,
        'scheduler': scheduler,
    })
    CONFIG.update({
        'task': args.task,
        'modeltype': args.modeltype,
        'n_structs': args.n_structs,
        'n_resamples': args.n_resamples,
        'steps': args.steps,
        'weight_decay': args.weight_decay,
        'save_every': args.save_every,
        'earlystop': args.earlystop,
        'patience': args.patience,
        'checkpoint': args.checkpoint,
        'graphclan': args.graphclan,
        'datafolder': args.datafolder,
        'outfolder': args.outfolder,
        'task': args.task, 
    })

    if args.modelname is None:
        now = datetime.datetime.now()
        CONFIG['modelname'] = f'{CONFIG["modeltype"]}_{args.modeltype}_{now.strftime("%m-%d-%Y-%H-%M-%S")}'
    else:
        CONFIG['modelname'] = args.modelname
    return CONFIG