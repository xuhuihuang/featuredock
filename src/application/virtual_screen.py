import argparse
import os
import sys
import time
import pickle
import copy
import itertools
from functools import partial
import multiprocessing
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from scipy import optimize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def read_result(predfile, voxelfile):
    with open(predfile, 'rb') as file:
        predictions = pickle.load(file)
    with open(voxelfile, 'rb') as file:
        coords = pickle.load(file)
    return coords, predictions


def load_ligand_fromsdf(ligfile):
    mol = Chem.MolFromMolFile(ligfile)
    return mol

def get_heavyatom(mol):
    pattern = Chem.MolFromSmarts("[!#1]")
    heavy_atom_matches = mol.GetSubstructMatches(pattern)
    heavy_atom_indices = list(itertools.chain(*heavy_atom_matches))
    heavy_atom_indices = sorted(set(list(heavy_atom_indices)))
    conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    return heavy_atom_indices, positions[heavy_atom_indices]

def random_shuffle(pos):
    t = np.random.randn(3).reshape(1, 3)
    r = R.from_rotvec(np.random.uniform(-np.pi, np.pi, 3))
    new_pos = (r.as_matrix()@(pos.T)).T + t
    return new_pos


def scoring_function(mol_coords, prob_coords, probabilities, cutoff=1.5):
    dist = cdist(mol_coords, prob_coords)
    dist_weighted = np.exp(-dist**2)*probabilities.reshape(1, -1)
    # average the squared differences to calculate the alignment
    mask = dist < cutoff
    score_per_atom = [np.mean(dist_weighted[i, :][mask[i, :]]) if any(mask[i, :]) else 0 for i in range(mask.shape[0])]
    mean_score = np.mean(score_per_atom)
    return mean_score


# define the objective function for the alignment
def objective_function(params, mol_coords, prob_coords, probabilities):
    """
    P(pi)*exp(-dist(pi-Tqi)**2)
    
    Params
    ------
    params : tuple
        6 elements, 
        the first three is rotation vector, 
        the last three is translational vector
    mol_coords : np.array
        shape of (M, 3)
    prob_coords : np.array
        shape of (N, 3)
    probabilities : np.array
        shape of (N, 1)

    Return
    ------
    score : float
    """
    # rotate the molecular structure using the rotation matrix
    rot_vec, trans_vec = params[:3], params[3:]
    rotation_matrix = R.from_rotvec(rot_vec)
    rotated_mol_coords = (rotation_matrix.as_matrix()@(mol_coords.T)).T + trans_vec.reshape(1, 3)
    dist = cdist(rotated_mol_coords, prob_coords)
    dist_weighted = np.exp(-dist**2)*probabilities.reshape(1, -1)
    mean_score = np.mean(dist_weighted)
    return -mean_score

def optimize_one_run_trimmed(orig_heavy_coords, prob_coords, prob_cutoff=0.5, dist_cutoff=1.5):
    heavy_coords = copy.deepcopy(orig_heavy_coords)
    mask = prob_coords[:, -1] > prob_cutoff
    prob_coords = prob_coords[mask, :]
    x0 = np.zeros(6)
    shuffled_heavy_coords = random_shuffle(heavy_coords)
    mol_center = np.mean(shuffled_heavy_coords, axis=0) # heavyatom center
    map_center = np.mean(prob_coords[prob_coords[:, -1]>prob_cutoff][:, :3], axis=0)
    heavy_coords_wo_mean = shuffled_heavy_coords - mol_center + map_center
    init_score = scoring_function(heavy_coords_wo_mean, prob_coords[:, :3], prob_coords[:, -1], dist_cutoff)
    func = partial(objective_function, mol_coords=heavy_coords_wo_mean, \
        prob_coords=prob_coords[:, :3], probabilities=prob_coords[:, -1])
    result = optimize.minimize(func, x0, bounds=None, method="L-BFGS-B", tol=1e-8)
    ## align based on result
    opt_params = result.x
    rot_vec, trans_vec = opt_params[:3], opt_params[3:]
    rotation_matrix = R.from_rotvec(rot_vec)
    aligned_mol_coords = (rotation_matrix.as_matrix()@(heavy_coords_wo_mean.T)).T + trans_vec.reshape(1, 3)
    opt_score = scoring_function(aligned_mol_coords, prob_coords[:, :3], prob_coords[:, -1], dist_cutoff)
    return result, init_score, heavy_coords_wo_mean, opt_score, aligned_mol_coords


def parse_args():
    parser = argparse.ArgumentParser(
        description='Soft align ligand to a predicted probability map'
    )
    parser.add_argument('--seed', type=int, default=0, \
        help='random seed')
    parser.add_argument('--voxelfile', type=str, \
        help='path to voxel coordinate file, shape of (N, 3)')
    parser.add_argument('--probfile', type=str, \
        help='path to probability file, shape of (N, 2)')
    parser.add_argument('--sdffile', type=str, \
        help='ligand structure in sdf format')
    parser.add_argument('--prob_cutoff', type=float, default=0.5, \
        help='voxels with predicted probability greater or equal to this value will be used in alignment')
    parser.add_argument('--outdir', type=str, \
        help='folder to store outputs')
    parser.add_argument('--embedding', default=False, action='store_true', \
        help='use random ligand conformations when flagged, ' \
            'otherwise use the provided conformer in sdffile')
    parser.add_argument('--nconfs', type=int, default=20, \
        help='Maximum number of ligand conformations to sample')
    parser.add_argument('--conf_rmsd', type=float, default=1.0, \
        help='RMSD cutoff for ligand conformations to be considered as similar confs')
    parser.add_argument('--nsamples', type=int, default=100, \
        help='Number of attemts for each ligand conformation')
    parser.add_argument('--threads', type=int, default=None, \
        help='Number of threads to use')
    parser.add_argument('--redoopt', action='store_true', default=False, \
        help='When flagged, overwrite exsiting optimization result file *results.pkl even if already exists')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    init_seed(args.seed)
    ncpus = multiprocessing.cpu_count() if args.threads is None else args.threads
    os.makedirs(args.outdir, exist_ok=True)
    ## sample engine
    ps = AllChem.ETKDGv3()
    ps.clearConfs=True
    ps.randomSeed=0xf00d
    ps.numThreads=ncpus
    ps.pruneRmsThresh=args.conf_rmsd
    ps.useExpTorsionAnglePrefs=True
    ps.useBasicKnowledge=True
    ## load data
    coords, predictions = read_result(args.probfile, args.voxelfile)
    PROB_COORDS = np.hstack([coords, predictions[:, -1].reshape((-1, 1))])
    print('Shape of probability coordinates:', PROB_COORDS.shape)
    ligname = os.path.splitext(os.path.basename(args.sdffile))[0]
    resultfile = os.path.join(args.outdir, \
                    f'{ligname}.{args.nconfs}confs.{args.nsamples}attempts.results.pkl')
    ## sampling and optimize
    if os.path.exists(resultfile) and (not args.redoopt):
        with open(resultfile, 'rb') as file:
            opt_results = pickle.load(file)
        print(f'Load optimized results from existing file: {resultfile}')
    else:
        if args.embedding:
            conffile = os.path.join(args.outdir, f'{ligname}.{args.nconfs}confs.sdf')
            lig = load_ligand_fromsdf(args.sdffile)
            lig = Chem.AddHs(lig)
            try:
                cids = AllChem.EmbedMultipleConfs(lig, args.nconfs, ps)
                for cid in cids: 
                    AllChem.MMFFOptimizeMolecule(lig, confId=cid)
            except Exception as e:
                print(f"Fail to process {args.sdffile}: e")
                sys.exit(1)
            writer = Chem.SDWriter(conffile)
            for cid in range(lig.GetNumConformers()):
                writer.write(lig, confId=cid)
        else:
            lig = load_ligand_fromsdf(args.sdffile)
        heavy_atom_indices, _ = get_heavyatom(lig)
        start = time.time()
        pool = multiprocessing.Pool(ncpus)
        func = partial(optimize_one_run_trimmed, prob_coords=PROB_COORDS, prob_cutoff=args.prob_cutoff, dist_cutoff=1.5)
        todolist = itertools.chain(*[[lig.GetConformer(id=cid).GetPositions()[heavy_atom_indices]]*args.nsamples \
            for cid in range(lig.GetNumConformers())])
        opt_results = pool.map(func, todolist)
        pool.close()
        pool.join()
        end = time.time()
        print(f'Time spent on optimizing for {ligname}: {lig.GetNumConformers()} ' \
            f'({args.nsamples} per conformer): {end-start} secs.')
        with open(resultfile, 'wb') as file:
            pickle.dump(opt_results, file)
