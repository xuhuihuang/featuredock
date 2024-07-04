import argparse
import numpy as np
import pickle
import os

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


def make_gridbox(xmin, xmax, ymin, ymax, zmin, zmax, spacing=1.0):
    """
    Parameters
    ----------
    pdbfile : str
        Filename of the apo protein
    xmin : float
    xmax : float
    ymin : float
    ymax : float
    zmin : float
    zmax : float
    spacing : float
        resolution of grid points
    outdir : str
        Path to the output folder
    
    Return
    ------
    lattice_tuple : tuple
        Each item in the tuple is a 3-d (x, y, z) coordinate
    """
    # voxelise the grid box
    x_numintervals = int(np.abs(xmax-xmin)/spacing)+1
    y_numintervals = int(np.abs(ymax-ymin)/spacing)+1
    z_numintervals = int(np.abs(zmax-zmin)/spacing)+1
    x = np.linspace(xmin, xmax, x_numintervals)
    y = np.linspace(ymin, ymax, y_numintervals)
    z = np.linspace(zmin, zmax, z_numintervals)
    xv, yv, zv= np.meshgrid(x, y, z)
    lattice_coords = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()])
    lattice_coords = lattice_coords.T # (N, 3)
    return lattice_coords


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Calculate FEATURE vectors for grid points in the pocket""")
    parser.add_argument('--name', type=str, default='query', \
        help='provide a name for the query; the output would be <name>_gridpoints.voxels.pkl')
    parser.add_argument('--xmin', type=float, \
        help='XMIN for the grid box')
    parser.add_argument('--xmax', type=float, \
        help='XMAX for the grid box')
    parser.add_argument('--ymin', type=float, \
        help='YMIN for the grid box')
    parser.add_argument('--ymax', type=float, \
        help='YMAX for the grid box')
    parser.add_argument('--zmin', type=float, \
        help='ZMIN for the grid box')
    parser.add_argument('--zmax', type=float, \
        help='ZMAX for the grid box')
    parser.add_argument('--outdir', type=str, \
        help='folder to store output files: <name>_gridpoints.voxels.pkl and <name>_gridpoints.xyz')
    parser.add_argument('--spacing', type=float, default=1.0, \
        help='resolution of grid points')
    parser.add_argument('--intermediate', action='store_true', \
        default=False, help='Write intermediate files such as xyz files when flagged.')
    args = parser.parse_args()

    name = args.name
    outdir = args.outdir
    
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    
    lattice_coords = make_gridbox(args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax, spacing=args.spacing)
    if args.intermediate:
        to_xyz_file(lattice_coords, os.path.join(outdir, f'{name}_gridpoints.xyz'))
    outfile = os.path.join(outdir, f'{name}_gridpoints.voxels.pkl')
    with open(outfile, 'wb') as file:
        pickle.dump(lattice_coords, file)