import pickle
import argparse
import os, random
import time
from sklearn.cluster import DBSCAN
# from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances
import numpy as np


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Soft align ligand to a predicted probability map'
    )
    parser.add_argument('--seed', type=int, default=0, \
        help='random seed')
    parser.add_argument('--resultfile', type=str, \
        help='result from soft alignment')
    parser.add_argument('--outfile', type=str, \
        help='file path to store outputs')
    parser.add_argument('--cls_stats', type=str, choices=['avg', 'max'], \
        default='avg', help='stats used in calculating cluster score')
    parser.add_argument('--cls_rmsd', type=float, default=1.0, \
        help='RMSD for optimized ligand poses to be considered as similar pose')
    parser.add_argument('--redogrp', action='store_true', default=False, \
        help='When flagged, overwrite exsiting group file *.cluster_results*.pkl even if already exists')
    args = parser.parse_args()
    return args

def group_results_dbscan(opt_results, method="avg", cls_rmsd=1.0):
    """
    1. Group screening results by rmsd.
    2. Sort groups based on average scoring function
    3. Pick the orientation with highest score in each group as the representative of the group
    4. select k best groups
    
    Params
    ------
    opt_results: lst
        Each item: result, init_score, heavy_coords_wo_mean, opt_score, aligned_mol_coords
    """
    scores = np.array([item[-2] for item in opt_results])
    X = [item[-1].reshape(-1) for item in opt_results]
    distances = pairwise_distances(X, metric=lambda x, y: np.sqrt(np.mean((x-y)**2)*3), \
        n_jobs=16, force_all_finite=True)
    scan = DBSCAN(eps=cls_rmsd, min_samples=1, metric='precomputed', \
        algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit(distances)
    if method == 'avg':
        groupby = [np.mean(scores[scan.labels_==i]) for i in range(len(np.unique(scan.labels_)))]
    elif method == 'max':
        groupby = [np.max(scores[scan.labels_==i]) for i in range(len(np.unique(scan.labels_)))]
    else:
        raise ValueError('Group score method not supported.')
    group_indices = np.argsort(groupby)[::-1]
    choices = []
    for i in group_indices:
        group = [opt_results[j] for j in range(len(opt_results)) if scan.labels_[j]==i]
        group = sorted(group, reverse=True, key=lambda x: x[-2])
        choices.append(group[0])
    return scan.labels_, group_indices, choices


if __name__ == '__main__':
    args = parse_args()
    init_seed(args.seed)
    resultfile = args.resultfile
    grp_file = args.outfile
    
    with open(resultfile, 'rb') as file:
        opt_results = pickle.load(file)
    print(f'Load optimized results from existing file: {resultfile}')
    if os.path.exists(grp_file) and (not args.redogrp):
        print(f'Clustered results already exists: {grp_file}')
    else:
        try:
            start = time.time()
            labels, group_indices, choices = group_results_dbscan(opt_results, method=args.cls_stats, cls_rmsd=args.cls_rmsd)
            end = time.time()
            print(f'Time spent on clustering predicted poses for {resultfile}: {end-start} secs.')
            with open(grp_file, 'wb') as file:
                pickle.dump((labels, group_indices, choices), file)
        except Exception as e:
            print(f"Clustering error: {resultfile}, {e}")