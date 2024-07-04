"""
Coordinates to be featurized: voxels/xxxx.voxel.pkl
Labels of features: voxels/xxxx.voxel.labels.pkl
"""
import os
import sys
import pickle
import argparse
import shutil
import subprocess
import numpy as np


def coord_to_ptf(pdbid, coords, ptffile):
    """
    Write coordinate to point file, 
    and point file is the input to FEATURE program.
    """

    with open(ptffile, 'w') as file:
        n = coords.shape[0]
        for i in range(n):
            file.write(f'{pdbid}\t{coords[i, 0]:.3f}\t{coords[i, 1]:.3f}\t{coords[i, 2]:.3f}\t#\tVoxel')
            file.write(os.linesep)

def ptf_to_ff(ptffile, searchpath, fffile, logfile, numshell, width, altman_dir):
    """
    Create FEATURE file for the points in ptffile, 
    using protein information stored in pdbfile and fffile.
    """
    try:
        cmd = f"export PDB_DIR={os.path.join(altman_dir, 'data', 'pdb')}; " \
                f"export DSSP_DIR={os.path.join(altman_dir, 'data', 'dssp')}; " \
                f"export FEATURE_DIR={os.path.join(altman_dir, 'data')}; " \
                f"{os.path.join(altman_dir, 'src')}/featurize " \
                f"-n {numshell} " \
                f"-w {width} " \
                f"-P {ptffile} " \
                f"-s {searchpath} > {fffile} 2> {logfile}"
        subprocess.Popen(cmd, shell=True).wait()
    except Exception as e:
        print(f'{ptffile}: Error Occurred: {e}')


def ff_to_pvar(fffile, pvarfile):
    """
    Store FEATURE file as a pickle file.
    """
    ffprop = []
    with open(fffile, 'r') as file:
        for line in file.readlines():
            if line.startswith('#'):
                # comment hearders
                continue
            prop = np.array(line.split("\t")[1:-6], dtype = np.float32)
            ffprop.append(prop)
    ## write to files
    ffprop = np.vstack(ffprop).astype(float) # convert properties to np.array
    with open(pvarfile, 'wb') as file:
        pickle.dump(ffprop, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Featurize grid point using Altman FEATURE vector")
    parser.add_argument('--pdbid', type=str, \
        help='PDB ID')
    parser.add_argument('--voxelfile', type=str, \
        help="""*.voxels.pkl""")
    parser.add_argument('--voxeldir', type=str, \
        help="""Folder that contains xxxx.voxels.pkl files
        (a numpy.ndarray with shape (N, 3)), 
        xxxx._interaction_.labels.pkl files 
        (a numpy.ndarray with indices, 
        xxxx.property.pvar files 
        (a numpy.ndarray with shape (N, nshell*80)), 
        """)
    parser.add_argument('--tempdir', type=str, \
        help="""Folder to store intermediate files during featurization, 
        such as xxxx.ptf file and xxxx.ff file.
        """)
    parser.add_argument('--searchdir', type=str, \
        help="""Search path used in featurize 
        which contains dssp file and apo pdb file, 
        e.g: ./data/apo/""")
    parser.add_argument('--featurize', type=str, default='./featurize-3.1.0', \
        help='path to featurize program.')
    parser.add_argument('--numshell', type=int, default=6, \
        help='path to featurize program.')
    parser.add_argument('--width', type=float, default=1.25, \
        help='path to featurize program.')
    parser.add_argument('--overwrite', default=False, action='store_true', \
        help='Whether to overwrite existing files')
    args = parser.parse_args()

    voxelfile = args.voxelfile
    ptffile = os.path.join(args.tempdir, f'{args.pdbid}.ptf')
    fffile = os.path.join(args.tempdir, f'{args.pdbid}.ff')
    logfile = os.path.join(args.tempdir, f'{args.pdbid}.log')
    pvarfile = os.path.join(args.voxeldir, f'{args.pdbid}.property.pvar')
    
    if (not args.overwrite) and (os.path.exists(pvarfile)):
        print(f'Property file already exists: {pvarfile}')
        sys.exit(1)
    with open(voxelfile, 'rb') as file:
        coords = pickle.load(file)
    coord_to_ptf(args.pdbid, coords, ptffile)
    ptf_to_ff(ptffile, args.searchdir, fffile, logfile, \
        args.numshell, args.width, args.featurize)
    ff_to_pvar(fffile, pvarfile)
    print(f"Finish featurizing {args.pdbid} voxels.")