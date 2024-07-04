import os
import sys
import pickle
import argparse
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import itertools
import shutil

SMARTS_PATTERNS = {
    "Heavyatom": ("[!#1]", ), 
    "Hydrophobic": ("[#6,#16,F,Cl,Br,I,At;+0]", ),
    "HBDonor": ("[#7,#8,#16][H]", ), 
    "HBAcceptor": ("[N,O,F,-{1-};!+{1-}]", ), 
    "Pi": ("a1:a:a:a:a:a:1", "a1:a:a:a:a:1"),
    "Cation": ("[+{1-}]", ),
    "Anion": ("[-{1-}]", )
}

def read_mol(molfile, filetype, removeHs=False):
    if not os.path.exists(molfile):
        print(f'Molecule file {molfile} does not exist.')
        return None
    with open(molfile, 'r') as file:
        content = file.read()
    if filetype == 'smi':
        mol = Chem.MolFromSmiles(content)
        AllChem.EmbedMolecule(mol)
    elif filetype == 'inchi':
        mol = Chem.MolFromInchi(content)
        AllChem.EmbedMolecule(mol)
    elif filetype == 'sdf':
        mol = Chem.MolFromMolBlock(content)
    else: 
        print('The given molecule format has not been supported yet.')
        return None
    return mol


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


def calc_interaction_maps(molname, mol, interactions, combineHB, outdir, intermediate=True):
    all_interactions = interactions
    allint_indices = {}
    for interaction in all_interactions:
        patterns = SMARTS_PATTERNS.get(interaction, tuple())
        allint_indices[interaction] = set()
        for pat in patterns:
            pattern = Chem.MolFromSmarts(pat)
            matches = mol.GetSubstructMatches(pattern)
            if interaction == "HBDonor":
                atom_idx = list(itertools.chain(*[(match[0], ) for match in matches]))
            else:
                atom_idx = list(itertools.chain(*matches))
            allint_indices[interaction] = allint_indices[interaction].union(atom_idx)
    if combineHB:
        allint_indices["HBond"] = allint_indices["HBDonor"].union(allint_indices["HBAcceptor"])
    for cid in range(mol.GetNumConformers()):
        allint_coords = {}
        conf_coords = mol.GetConformer(cid).GetPositions()
        for interaction, indices in allint_indices.items():
            sort_idx = sorted(indices)
            allint_coords[interaction] = conf_coords[sort_idx, :]
        if intermediate:
            subdir = os.path.join(outdir, f'{molname}_conf{cid}')
            if not os.path.exists(subdir):
                os.makedirs(subdir, exist_ok=True)
                print(f'Create folder: {subdir}')
            for interaction, coords in allint_coords.items():
                xyzfile = os.path.join(subdir, f'{molname}_{interaction}.xyz')
                to_xyz_file(coords, xyzfile)
        ## output interaction maps
        outfile = os.path.join(outdir, f'{molname}_conf{cid}.intmap')
        with open(outfile, 'wb') as file:
            pickle.dump(allint_coords, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Convert molecule to point maps.
    """)
    parser.add_argument('--molfile', type=str, \
        help='path to molecule file')
    parser.add_argument('--filetype', type=str, default='sdf', \
        choices=['smi', 'inchi', 'sdf'])
    parser.add_argument('--embed', type=int, default=10, \
        help='Number of 3D embeddings for the molecule structure')
    parser.add_argument('--interactions', type=str, nargs='+', \
        help='e.g: Heavyatom, Hydrophobic, HBDonor, HBAcceptor, Pi, Cation, ...')
    parser.add_argument('--combineHB', action='store_true', default=False, \
        help='Combine HBAcceptor and HBDonor positions if flagged')
    parser.add_argument('--intermediate', action='store_true', default=False, \
        help='Output configurations to xyz files if flagged')
    parser.add_argument('--outdir', type=str, \
        help='output folder')
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
        print(f'Create folder {args.outdir}')
    molname, _ = os.path.splitext(os.path.basename(args.molfile))
    mol = read_mol(args.molfile, args.filetype)
    if mol is None:
        print('Fail to read molecule structure.')
        sys.exit(1)
    ## embed 3D configurations and write to file
    if args.embed > 1:
        mol = Chem.AddHs(mol)
        confs = AllChem.EmbedMultipleConfs(mol, numConfs=args.embed)
        confdir = os.path.join(args.outdir, molname+f'_{args.embed}confs')
    else:
        args.embed = 0
        confdir = os.path.join(args.outdir, molname+f'_original_conf')
    
    if not os.path.exists(confdir):
        os.makedirs(confdir, exist_ok=True)
        print(f'Create folder {confdir}')
    for cid in range(mol.GetNumConformers()):
        conffile = os.path.join(confdir, molname+f'_conf{cid}.sdf')
        writer = Chem.SDWriter(conffile)
        writer.write(mol, confId=cid)
    ## calculate interaction maps
    calc_interaction_maps(molname, mol, args.interactions, \
        args.combineHB, confdir, args.intermediate)
    