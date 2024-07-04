"""
Extract proximal protein sites to ligands in co-crystal PDB structures;
Annotate protein-ligand interactions using ProLIG;
Write annotation to files.


References:
Example code LigPlot for PPI: https://github.com/chemosim-lab/ProLIF/issues/58
"""
import sys
from collections import defaultdict
import itertools
import os
import argparse
import numpy as np
import rdkit.Chem as Chem
from scipy import spatial
import pickle
import json

def to_xyz_file(coords, outfile):
    """
    Parameters
    ----------
    coords : numpy.ndarray
        Shape of (N, 3), each line represents a (x, y, z) coordinate
    outfile : str
        Path to the output file
    
    Return
    ------
    None
    """
    n = coords.shape[0]
    pattern = '{label}\t{x:.3f}\t{y:.3f}\t{z:.3f}\n'
    with open(outfile, 'w') as file:
        for i in range(n):
            file.write(pattern.format(label='H', x=coords[i][0], \
                y=coords[i][1], z=coords[i][2]))
    return


def label_voxels(name, interaction, lattice_coords, landmark_coord, \
            cutoff, outdir, intermediate=False):
    """
    Parameters
    ----------
    name : str
        Name for output files
    interaction : str
        'HeavyAtomsite', 'Hydrophobic', ...
    lattice_coords : numpy.ndarray
        Shape of (M, 3)
        If grid points are trimmed in function "voxelise",
        shape of the lattice is not cuboid.
    landmark_coord : (N, 3)
        [(x1, y1, z1), (x2, y2, z2)], ...
        Positions of ligand atoms
    cutoff : float
        Voxels within the cutoff can be considered as the same type
        as the landmark. For example, cutoff=1.8 is the diagonal 
        distance of a 1*1*1 grid box.
        {'HeavyAtomsite': 1.8, 'Hydrophobic': 1.8, ...}
    outdir : str
        Path to store *.{interaction}.labels.pkl, *.{interaction}.labels.xyz
    intermediate: bool
        Whether to store *.{interaction}.labels.xyz

    Return
    ------
    labels : np.array
        Shape of (M, 2)
        Indices and labels of voxels
        Indices range from [0, M), labels are either 0 or 1
    """
    landmark_tree = spatial.cKDTree(landmark_coord)
    landmark_nearest_dist, landmark_nearest = landmark_tree.query(
                                    lattice_coords, k=[1],
                                    eps=0, p=2,
                                    distance_upper_bound=np.inf,
                                    workers=1
                                )
    landmark_nearest_dist = landmark_nearest_dist.T[0]
    landmark_nearest = landmark_nearest.T[0]
    index_within = np.where(
                    landmark_nearest_dist < cutoff
                )[0].tolist()
    labels = np.zeros((lattice_coords.shape[0], 2))
    labels[:, 0] = np.arange(lattice_coords.shape[0])
    labels[index_within, 1] = 1
    if intermediate:
        coords = lattice_coords[index_within]
        to_xyz_file(coords, os.path.join(outdir, f'{name}.{interaction}.labels.xyz'))
    outfile = os.path.join(outdir, f'{name}.{interaction}.labels.pkl')
    with open(outfile, 'wb') as file:
        pickle.dump(labels, file)
    return labels

def soft_label_voxels(name, interaction, lattice_coords, landmark_coord, \
            cutoffs, outdir, intermediate=False):
    """
    name : str
        Name for output files
    interaction : str
        'HeavyAtomsite', 'Hydrophobic', ...
    lattice_coords : numpy.ndarray
        Shape of (M, 3)
        If grid points are trimmed in function "voxelise",
        shape of the lattice is not cuboid.
    landmark_coord : (N, 3)
        [(x1, y1, z1), (x2, y2, z2)], ...
        Positions of ligand atoms
    cutoffs : np.array
        cutoffs = [[0.0, 1.0, 1.5, 1.8, 2.0], [1.0, 1.0, 0.7, 0.3, 0.0, 0.0]]
        bins, factors = cutoffs[0], cutoffs[1]
        Voxels within the cutoff can be considered as the same type
        as the landmark. But voxels with a higher distance will
        be labeled partially positive instead of totally positive.
    outdir : str
        Path to store *.{interaction}.labels.pkl, *.{interaction}.labels.xyz
    intermediate: bool
        Whether to store *.{interaction}.labels.xyz
    
    Return
    ------
    partial_values : np.array
        Shape of (M, 2)
        Indices and labels of voxels
        Indices range from [0, M), labels range from [0, 1].
    """
    bins, factors = cutoffs[0], cutoffs[1]
    landmark_tree = spatial.cKDTree(landmark_coord)
    landmark_nearest_dist, landmark_nearest = landmark_tree.query(
                                    lattice_coords, k=[1], 
                                    eps=0, p=2, 
                                    distance_upper_bound=np.inf, 
                                    workers=1
                                )
    landmark_nearest_dist = landmark_nearest_dist.T[0]
    partial_values = np.hstack([np.arange(lattice_coords.shape[0]).reshape(-1, 1), 
        np.ones(lattice_coords.shape[0]).reshape(-1, 1)])
    factor_indices = np.digitize(landmark_nearest_dist, bins, right=False)
    partial_values[:, 1] = partial_values[:, 1]*np.array(factors)[factor_indices]
    if intermediate:
        coords = lattice_coords[np.nonzero(partial_values[:, 1])[0]]
        to_xyz_file(coords, os.path.join(outdir, f'{name}.{interaction}.smoothed.labels.xyz'))
    outfile = os.path.join(outdir, f'{name}.{interaction}.smoothed.labels.pkl')
    with open(outfile, 'wb') as file:
        pickle.dump(partial_values, file)
    return partial_values

def label_charges(name, interaction, lattice_coords, landmark_info, \
            cutoffs, outdir, intermediate=False):
    raise NotImplementedError("Partial charge dataset not implemented.")
# def label_charges(name, interaction, lattice_coords, landmark_info, \
#             cutoffs, outdir, intermediate=False):
#     """
#     name : str
#         Name for output files
#     interaction : str
#         'HeavyAtomsite', 'Hydrophobic', ...
#     lattice_coords : numpy.ndarray
#         Shape of (M, 3)
#         If grid points are trimmed in function "voxelise",
#         shape of the lattice is not cuboid.
#     landmark_info : (N, 4)
#         [(x1, y1, z1, v1), (x2, y2, z2, v2)], ...
#         Positions of ligand atoms and values of the position
#     cutoffs : 
#         Voxels within the cutoff can be considered as the same type
#         as the landmark. For example, cutoff=1.8 is the diagonal 
#         distance of a 1*1*1 grid box.
#     outdir : str
#         Path to store *.{interaction}.labels.pkl, *.{interaction}.labels.xyz
#     intermediate: bool
#         Whether to store *.{interaction}.labels.xyz
    
#     Return
#     ------
#     labels : np.array
#         Shape of (M, 2)
#         Indices and labels of voxels
#         Indices range from [0, M), values are smoothed partial charges
#     """
#     landmark_coord, landmark_val = landmark_info[:, :3], landmark_info[:, -1]
#     landmark_tree = spatial.cKDTree(landmark_coord)
#     landmark_nearest_dist, landmark_nearest = landmark_tree.query(
#                                     lattice_coords, k=[1], 
#                                     eps=0, p=2, 
#                                     distance_upper_bound=np.inf, 
#                                     workers=1
#                                 )
#     landmark_nearest_dist = landmark_nearest_dist.T[0]
#     landmark_nearest = landmark_nearest.T[0]
#     partial_values = np.vstack([np.arange(lattice_coords.shape[0]).reshape(-1, 1), 
#         np.ones(lattice_coords.shape[0]).reshape(-1, 1)])
#     factor_indices = np.digitize(landmark_nearest_dist, bins, right=False)
#     partial_values[:, 1] = landmark_val[landmark_nearest]*factors[factor_indices]
#     if intermediate:
#         coords = lattice_coords[index_within]
#         to_xyz_file(coords, os.path.join(outdir, f'{name}.{interaction}.labels.xyz'))
#     outfile = os.path.join(outdir, f'{name}.{interaction}.labels.pkl')
#     with open(outfile, 'wb') as file:
#         pickle.dump(index_within, file)
#     return index_within


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Voxelize and label potential binding pocket.
    """)
    parser.add_argument('--pdbid', type=str, help='PDBID')
    parser.add_argument('--voxelfile', type=str, \
        help='voxel pickle file and landmark pickle files')
    parser.add_argument('--lmfile', type=str, \
        help='voxel pickle file and landmark pickle files')
    parser.add_argument('--outdir', type=str, \
        help='folder to store output files')
    parser.add_argument('--configfile', type=str, default='label_config.json', \
        help='config file containing interaction types, cutoffs, etc.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--hard', action="store_true", default=False, \
        help="Use hard labeling, voxels will be labeled as either positive(1) or negative(0)")
    group.add_argument('--soft', action="store_true", default=False, \
        help="Use soft labeling, voxels will be labeled smoothly, such as 1.0, 0.5, 0.0, etc.")
    parser.add_argument('--interactions', type=str, nargs='+', default=None, \
        choices=["HeavyAtomsite", "Chargesite", \
            "Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking", \
            "Anionic", "Cationic", "CationPi", "PiCation"], \
        help='protein-ligand interactions to consider')
    parser.add_argument('--intermediate', action='store_true', \
        default=False, help='Write intermediate files such as xyz files when flagged.')
    parser.add_argument('--overwrite', action='store_true', \
        default=False, help='Redo the calculation and overwrite existing file when flagged.')
    args = parser.parse_args()
    
    voxelfile = args.voxelfile
    landmark_file = args.lmfile
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
        print(f'Create output folder {args.outdir}.')

    ## load files
    with open(args.configfile, 'r') as file:
        config = json.load(file)
    if args.hard:
        select_config = config["hard_cutoffs"]
        label_method = label_voxels
    elif args.soft:
        select_config = config["soft_cutoffs"]
        label_method = soft_label_voxels
    else:
        raise ValueError("Please specify one labeling strategy: hard (0/1) or soft (0/0.3/0.7/1.0)")
    ## fetch grid points aka voxels, and landmark files
    if not os.path.exists(voxelfile):
        print(f'Voxel file does not exist: {voxelfile}.')
        sys.exit(1)
    if not os.path.exists(landmark_file):
        print(f'Landmark file does not exist: {landmark_file}.')
        sys.exit(1)
    with open(voxelfile, 'rb') as file:
        lattice_coords = pickle.load(file)
    with open(landmark_file, 'rb') as file:
        landmarks = pickle.load(file)

    ## label voxels
    for interaction in args.interactions:
        labeld_file = os.path.join(args.outdir, f'{args.pdbid}.{interaction}.labels.pkl')
        if interaction not in landmarks.keys():
            continue
        if os.path.exists(labeld_file) and (not args.overwrite):
            print(f'Voxels have already been labeled: {labeld_file}')
            continue
        else:
            if interaction == 'Chargesite':
                labeled = label_charges(args.pdbid, interaction, lattice_coords, \
                                landmarks[interaction], select_config[interaction], \
                                args.outdir, intermediate=args.intermediate)
            else:
                labeled = label_method(args.pdbid, interaction, lattice_coords, \
                                landmarks[interaction], select_config[interaction], \
                                args.outdir, intermediate=args.intermediate)
    print("Succesfully voxelise and label structure:", args.pdbid)