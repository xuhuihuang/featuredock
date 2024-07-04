import argparse
import os
import sys
import pickle
import itertools
import rdkit.Chem as Chem
from rdkit.Geometry import Point3D


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='plot aligned results'
    )
    parser.add_argument('--grp_file', type=str, \
        help='path to clustered results')
    parser.add_argument('--sdffile', type=str, \
        help='path to compound structure')
    parser.add_argument('--topK', type=int, default=10, \
        help='Plot represenattive poses of top-k clusters')
    parser.add_argument('--outdir', type=str, \
        help='folder to store output results')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    grp_file = args.grp_file
    if os.path.exists(grp_file):
        with open(grp_file, 'rb') as file:
            labels, group_indices, choices = pickle.load(file)
    else:
        print(f'{grp_file} does not exist')
        sys.exit(1)

    lig = load_ligand_fromsdf(args.sdffile)
    heavy_idx, coords = get_heavyatom(lig)
    conf = lig.GetConformer()
    for confid in range(args.topK):
        try:
            w = Chem.SDWriter(os.path.join(args.outdir, f'top{confid}.vsaligned.sdf'))
            aligned_mol_coords = choices[confid][-1]
            for i,j in enumerate(heavy_idx):
                x,y,z = aligned_mol_coords[i]
                conf.SetAtomPosition(j, Point3D(x,y,z))
            w.write(lig)
            w.close()
        except Exception as e:
            print(f"Error occurred when processing Conformer {confid} of {grp_file}.")
            break