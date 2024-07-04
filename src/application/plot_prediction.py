import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.linewidth'] = 2
linewidth = 2


def read_result(predfile, voxelfile):
    with open(predfile, 'rb') as file:
        predictions = pickle.load(file)
    with open(voxelfile, 'rb') as file:
        coords = pickle.load(file)
    return coords, predictions


def visualise(coords, predictions):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.xaxis.set_tick_params(which='major', size=10, width=linewidth, direction='out')
    ax.yaxis.set_tick_params(which='major', size=10, width=linewidth, direction='out')
    cmap = plt.get_cmap('RdBu_r', 10)
    ## plot density/probability map
    p = ax.scatter(coords[:,0], coords[:,1], coords[:,2], \
        c=predictions[:, 1], cmap=cmap, s=1.0, label='Probability Map')
    plt.legend(loc='upper left')
    plt.colorbar(p)
    return fig


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


def voxels_above(coords, predictions, cutoff, outfile):
    mask = predictions[:, 1]>=cutoff
    indices = np.where(mask==1)[0]
    pred_coords = coords[indices, :]
    to_xyz_file(pred_coords, outfile)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualise grid points and predicted probabilities'
    )
    parser.add_argument('--voxelfile', type=str, \
        help='path to voxel coordinates: shape of (N, 3)')
    parser.add_argument('--probfile', type=str, \
        help='path to predicted probabilities: shape of (N, 2)')
    parser.add_argument('--cutoffs', type=float, nargs='+', \
        default=[0.7, 0.8, 0.9, 0.95], help='probability cutoffs')
    parser.add_argument('--outdir', type=str, \
        default='folder to store output xyz files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    coords, predictions = read_result(args.probfile, args.voxelfile)
    for cutoff in args.cutoffs:
        xyzfile = os.path.join(args.outdir, f'cutoff{cutoff}.xyz')
        voxels_above(coords, predictions, cutoff, xyzfile)
    print(f'Finish writing xyz files above {args.cutoffs} for ' \
          f'({args.voxelfile}, {args.probfile})')