import os
import pickle
import argparse
import pandas as pd
import numpy as np
import re
import copy
import itertools


def readEntryList(filename, sep='[\s,; ]+'):
    """
    read a text file contains a list of objects ( ligand names / pdb ids / pubchem ids ), 
    which are seperated by common delimiter ( \tab, \space, comma, etc. )
    """
    with open(filename, 'r') as file:
        content = file.read()
    results = re.split(sep, content)
    results = [r.strip().lower() for r in results]
    return results

def assign_kfold(_clanids, n_splits):
    n_samples = len(_clanids)
    clanids = copy.deepcopy(_clanids)
    np.random.shuffle(clanids)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    foldids = np.full(n_samples, -1, dtype=int)
    current = 0
    for fid in range(n_splits):
        fold_size = fold_sizes[fid]
        start, stop = current, current + fold_size
        foldids[start:stop] = fid
        current = stop
    return dict(zip(clanids, foldids))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Split structure clusters into k-folds.
    """)
    parser.add_argument('--clanfile', type=str, \
        help="""pickle file containing a pd.DataFram,
            including clan_id and PDBList, 
            e.g: ClanGraph_90_df.pkl,
            """)
    parser.add_argument('--name', type=str, default='default', \
        help='assign a dataset name')
    parser.add_argument('--ignorefile', type=str, default=None, \
        help='file contains PDBIDs to remove')
    parser.add_argument('--k', type=int, default=10, \
        help='k-fold')
    parser.add_argument('--sqid', type=int, default=90, \
        help='sequence cluster cutoff, e.g: 30, 30, 50, 70, 90, 95, ...')
    parser.add_argument('--seed', type=int, default=42, \
        help='random seed, for reproduce purpose')
    parser.add_argument('--outdir', type=str, \
        help='folder that contains structure cluster dataframe, and to store k-fold split result')
    args = parser.parse_args()
    np.random.seed(args.seed)
    with open(args.clanfile, 'rb') as file:
        df = pickle.load(file)
    df['StructNumber'] = df.apply(lambda x: len(x['PDBIDList']), axis=1)
    # remove clusters that have shared structures with the PDBIDs to remove
    if args.ignorefile is not None:
        target_ids = readEntryList(args.ignorefile)
        target_ids = set(target_ids)
        print(f"Number of PDB IDs in ignore file: {len(target_ids)}.")
    else:
        target_ids = []
    # if the structure clan has shared target homologous structure -> used in training dataset
    # otherwise used in test
    df['IsRemoved'] = df.apply(lambda x: len(set(x['PDBIDList']).intersection(target_ids))>0, axis=1)
    print(f"Number of structure clans to remove: {df[df['IsRemoved']==True].shape[0]}")
    removed = itertools.chain(*df[df['IsRemoved']==True]['PDBIDList'].tolist())
    removed = list(removed)
    print(f"Number of structures removed: {len(removed)}")
    print(f"Structures removed: {removed}")
    # split the remaining structure clans to K-Fold
    clanids = df[df['IsRemoved']==False]['Structure_Clan_ID'].values.tolist()
    dct = assign_kfold(clanids, args.k)
    df['CrossFold'] = df.apply(lambda x: dct.get(x['Structure_Clan_ID'], -1), axis=1)
    outfile = os.path.join(args.outdir, f'ClanGraph_{args.name}_{args.sqid}_{args.k}Folds.pkl')
    with open(outfile, 'wb') as file:
        pickle.dump(df, file)