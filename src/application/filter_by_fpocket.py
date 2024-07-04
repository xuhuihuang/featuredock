"""
This script is used to filter grid points in query box
based on fpocket results (*.pqr) file.

python src/application/filter_by_fpocket.py \
    --voxelfile=${VOXEL_DIR}/ref.voxels.pkl \
    --pqrfile=${VOXEL_DIR}/pocket_1.pqr \
    --radius=5.0 \
    --outvoxel=${VOXEL_DIR}/ref.filtered.voxels.pkl \
    --intermediate
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from scipy import spatial

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


def load_vertices(pqrfile):
    lst = []
    with open(pqrfile, 'r') as file:
        for line in file.readlines():
            if line.startswith('ATOM'):
                lst.append(line.split())
    df = pd.DataFrame(lst, \
            columns=['field', 'atomid', 'element', 'resname', 'resid', 'x', 'y', 'z', 'e', 'r'])
    return df


def filter_points(coords, vertices, radius):
    vertices = vertices[:, :3]
    poc_tree = spatial.cKDTree(vertices)
    nearest_dist, nearest = poc_tree.query(coords, k=[1], eps=0, p=2, 
                                    distance_upper_bound=np.inf, workers=1)
    nearest_dist = nearest_dist.T[0]
    nearest = nearest.T[0]
    mask = nearest_dist <= radius
    filtered_coords = coords[mask]
    return filtered_coords

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--voxelfile', type=str, \
            help='*.pkl file that contains 3d coordinates, shape of (N, 3)')
    parser.add_argument('--pqrfile', type=str, \
            help='fpocket result')
    parser.add_argument('--radius', type=float, default=5.0, \
            help='distance cutoff between grid points and fpocket vertices')
    parser.add_argument('--outdir', type=str, \
            help='folder to output *.pkl file and intermediate *.xyz file')
    parser.add_argument('--intermediate', default=False, action='store_true', \
            help='output filtered grid points as xyz file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ## load fpocket alpha spheres
    df = load_vertices(args.pqrfile)
    vertices = df[['x', 'y', 'z']].values
    ## load original grid 
    with open(args.voxelfile, 'rb') as file:
        coords = pickle.load(file)
    filtered_coords = filter_points(coords, vertices, args.radius)
    ## write results
    name = os.path.basename(args.voxelfile)
    outfile = os.path.join(args.outdir, name)
    if args.intermediate:
        pid = name.split('.')[0]
        xyzfile = os.path.join(args.outdir, f'{pid}_gridpoints.xyz')
        to_xyz_file(filtered_coords, xyzfile)
    with open(outfile, 'wb') as file:
        pickle.dump(filtered_coords, file)

