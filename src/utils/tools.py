import os
import re
import random
import numpy as np
import torch


def make_dir(dest_dir):
    """Create folder recursively

    Parameters
    ----------
    dest_dir : str
        path of folder to create
    """
    os.makedirs(dest_dir, exist_ok=True)


def readEntryList(filename, sep='[\s,; ]+'):
    """Read from a text file containing a list of objects ( ligand names / pdb ids / pubchem ids ), 
    which are seperated by common delimiter ( \tab, \space, comma, etc.).
    Return the list of objects.

    Parameters
    ----------
    filename : str
        text file
    sep : str, optional
        delimiter pattern, by default '[\s,; ]+'

    Returns
    -------
    results : list
        list of objects stored in the file
    """
    with open(filename, 'r') as file:
        content = file.read()
    results = re.split(sep, content)
    results = [r.strip() for r in results]
    return results


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