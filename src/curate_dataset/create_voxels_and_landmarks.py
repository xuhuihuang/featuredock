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
from rdkit.Chem import rdEHTTools
dirname = os.path.dirname(os.path.abspath(__file__)) # curate_dataset
pardirname = os.path.dirname(dirname) # src
sys.path.append(pardirname)
import core_prolif as plf
from core_prolif import get_residues_near_ligand
from scipy import spatial
import pickle
from scipy.spatial import ConvexHull


def get_points_inside_hull(query_coords, hull):
    # The hull is defined as all points x for which Ax + b <= 0.
    # We compare to a small positive value to account for floating
    # point issues.
    # Assuming x is shape (m, d), output is boolean shape (m,).
    # A is shape (f, d) and b is shape (f, 1).
    A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    eps = np.finfo(np.float32).eps
    mask = np.all(query_coords@A.T + b.T < eps, axis=1)
    return mask


def to_pdb_file(outfile, *mols):
    """
    Parameters
    ----------
    outfile : str
        Path to the output file
    *mols : iterable
        A list of molecules (prolif.Molecule),
        which inherits from Chem.Mol

    Return
    ------
    None
    """
    with open(outfile, 'w') as file:
        for mol in mols:
            block = Chem.MolToPDBBlock(mol)
            file.write(block)
    return


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


def voxelise(name, lig, prot, outdir, \
        cutoff, spacing=1, trim=True, trim_cutoff=(1, 6), \
        abs_include=True, intermediate=False):
    """
    Parameters
    ----------
    name : str
        Name for output files
    lig : prolif.Molecule
    prot : prolif.Molecule
    cutoff : [float, float]
        voxels to pick around lig, inner cutoff & outer cutoff
    trim : boolean
        Whether to trim voxels outside the polygon,
        which is defined by atoms in proximal_resids.
    abs_include : boolean
        Whether to create grid box that covers whole ligand
        when the ligand is partially outside the binding pocket
        (defined by proximal_resids).
    Return
    ------
    lattice_tuple : tuple
        Each item in the tuple is a 3-d (x, y, z) coordinate
    """
    all_voxels = None
    for lresid, lres in lig.residues.items():
        proximal_resids = get_residues_near_ligand(lres, prot, cutoff=cutoff)
        prot_ress = [prot[prot_key] for prot_key in proximal_resids if prot_key.name!='HOH']
        res_coords = np.vstack([res.xyz for res in prot_ress])
        if abs_include:
            # in case ligand falls out of the pocket
            pock_coords = np.vstack([res_coords, lres.xyz])
        else:
            # grid box around the residues
            pock_coords = res_coords
        minx, maxx = np.min(pock_coords[:, 0]), np.max(pock_coords[:, 0])
        miny, maxy = np.min(pock_coords[:, 1]), np.max(pock_coords[:, 1])
        minz, maxz = np.min(pock_coords[:, 2]), np.max(pock_coords[:, 2])
        # voxelise the grid box
        x_numintervals = int(np.abs(maxx-minx)/spacing)+1
        y_numintervals = int(np.abs(maxy-miny)/spacing)+1
        z_numintervals = int(np.abs(maxz-minz)/spacing)+1
        x = np.linspace(minx, maxx, x_numintervals)
        y = np.linspace(miny, maxy, y_numintervals)
        z = np.linspace(minz, maxz, z_numintervals)
        xv, yv, zv= np.meshgrid(x, y, z)
        lattice_coords = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()])
        lattice_coords = lattice_coords.T # (N, 3)
        if trim:
            hull = ConvexHull(pock_coords)
            mask = get_points_inside_hull(lattice_coords, hull)
            res_tree = spatial.cKDTree(prot.xyz)
            nearest_dist, nearest = res_tree.query(
                                            lattice_coords, k=[1], 
                                            eps=0, p=2, 
                                            distance_upper_bound=np.inf, 
                                            workers=1
                                        )
            nearest_dist = nearest_dist.T[0]
            nearest = nearest.T[0]
            halo_mask = (nearest_dist >= trim_cutoff[0]) & (nearest_dist <= trim_cutoff[1])
            lattice_coords = lattice_coords[mask&halo_mask]
        if intermediate:
            to_xyz_file(lattice_coords, os.path.join(outdir, f'{name}_{lresid}.xyz'))
            to_pdb_file(os.path.join(outdir, f'{name}_{lresid}_proximal.pdb'), *prot_ress)
        if all_voxels is None:
            all_voxels = lattice_coords
        else:
            all_voxels = np.vstack([all_voxels, lattice_coords])
    outfile = os.path.join(outdir, f'{name}.voxels.pkl')
    with open(outfile, 'wb') as file:
        pickle.dump(all_voxels, file)
    return all_voxels


def create_interaction_landmarks(ifp, lig):
    """
    Parameters
    ----------
    ifp : dict
        See prolif.Fingerprint.generate
    
    Return
    ------
    landmarks_pos : dict
        {interaction_label1: [(x1, y1, z1), (x2, y2, z2)], }
        Interaction Type : Positions of ligand atoms
    """
    interactions = list(fp.interactions.keys())
    n_types = len(interactions)
    landmarks = {i: defaultdict(set) for i in interactions}
    data = ifp
    for res_pair, return_info in data.items():
        # interaction masks, index of ligand atom, index of protein atom
        lresid, presid = res_pair
        int_mask, lidx, pidx = return_info
        for i in range(n_types):
            if int_mask[i]:
                atom_idx = lidx[i]
                landmarks[interactions[i]][lresid].update(atom_idx)
    landmarks_pos = {}
    for interaction in landmarks.keys():
        if len(landmarks[interaction].keys())==0:
            continue
        landmarks_pos[interaction] = np.vstack([lig[lresid].xyz[sorted(idx)] \
            for lresid, idx in landmarks[interaction].items() if len(idx)!=0])
    return landmarks_pos


def create_heavy_atom_landmarks(lig):
    """
    Label heavy atom (non-Hydrogen atom) positions.

    Parameters
    ----------
    lig : ResidueGroup
    
    Return
    ------
    atom_landmarks : dict
        {'HeavyAtomsite': [(x1, y1, z1), (x2, y2, z2)], }
    """
    pattern = Chem.MolFromSmarts("[!#1]")
    heavy_coords = None
    for lresid, lres in lig.residues.items():
        lres_coords = lres.xyz
        heavy_atom_idx = itertools.chain(*lres.GetSubstructMatches(pattern))
        heavy_atom_idx = sorted(set(list(heavy_atom_idx)))
        pick_coords = lres_coords[heavy_atom_idx, :]
        if heavy_coords is None:
            heavy_coords = pick_coords
        else:
            heavy_coords = np.vtsack([heavy_coords, pick_coords])
    atom_landmarks = {'HeavyAtomsite': heavy_coords}
    return atom_landmarks


def create_partial_charge_landmarks(lig):
    """
    Label partial charges and positions.
    Reference:
        https://iwatobipen.wordpress.com/2020/01/08/partial-charge-visualization-with-rdkit-rdkit-quantumchemistry-psikit/
    Parameters
    ----------
    lig : ResidueGroup
    
    Return
    ------
    charge_landmarks : dict
        {'Chargesite': [(x1, y1, z1, q1), (x2, y2, z2, q2)], }
    """
    charges = None
    for lresid, lres in lig.residues.items():
        lres_coords = lres.xyz
        _, res = rdEHTTools.RunMol(lig)
        static_chgs = res.GetAtomicCharges()
        if charges is None:
            charges = np.hstack([lres_coords, static_chgs.reshape(-1, 1)])
        else:
            tmp = np.hstack([lres_coords, static_chgs.reshape(-1, 1)])
            charges = np.vstack([charges, tmp])
    charge_landmarks = {'Chargesite': charges}
    return charge_landmarks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Voxelize and label potential binding pocket.
    """)
    parser.add_argument('--pdbid', type=str, help='PDBID')
    parser.add_argument('--apofile', type=str, help='Fixed and cleaned apo protein')
    parser.add_argument('--hetfile', type=str, help='Fixed and cleaned heterogen')
    parser.add_argument('--outdir', type=str, \
        help='folder to store output files')
    parser.add_argument('--pocket_cutoff', type=float, default=6.0, \
        help='pocket size around the ligand')
    parser.add_argument('--spacing', type=float, default=1.0, \
        help='grid spacing interval')
    parser.add_argument('--trim', action='store_true', default=False, \
        help='trim down the number of voxels based on distance cutoff when flagged')
    parser.add_argument('--trim_min', type=float, default=0.0, \
        help='voxels above this cutoff away from the protein atoms will be kept')
    parser.add_argument('--trim_max', type=float, default=6.0, \
        help='voxels below this cutoff away from the protein atoms will be kept')
    parser.add_argument('--abs_include', action='store_true', default=False, \
        help='')
    parser.add_argument('--interactions', type=str, nargs='+', default=None, \
        choices=["Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking", \
            "Anionic", "Cationic", "CationPi", "PiCation"], 
        help='protein-ligand interactions to consider')
    parser.add_argument('--heavyatom', action='store_true', \
        default=False, help='Create heavyatom landmarks')
    parser.add_argument('--charge', action='store_true', \
        default=False, help='Create charge landmarks')
    parser.add_argument('--intermediate', action='store_true', \
        default=False, help='Write intermediate files such as xyz files when flagged.')

    parser.add_argument('--overwrite', action='store_true', \
        default=False, help='Redo the calculation and overwrite existing file when flagged.')
    args = parser.parse_args()

    apofile = args.apofile
    hetfile = args.hetfile
    trim_cutoff = [args.trim_min, args.trim_max]
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
        print(f'Create output folder {args.outdir}.')

    ## load cocrystal molecule
    try:
        prot = Chem.MolFromPDBFile(apofile, removeHs=False)
        lig = Chem.MolFromMolFile(hetfile, removeHs=False)
    except Exception as e:
        print("Error occurred:", args.pdbid)
        print(e)
        sys.exit(1)
    
    ## create plf.Molecule
    try:
        prot = plf.Molecule(prot)
        lig = plf.Molecule.from_rdkit(lig)
        if lig.n_residues != 1:
            raise NotImplementedError(f"""
                Methods for dealing with protein structure containing 
                multiple ligands have not been supported: {args}.
            """)
    except Exception as e:
        print("Error occurred:", args.pdbid)
        print(e)
        sys.exit(1)

    ## fetch grid points
    voxelfile = os.path.join(args.outdir, f'{args.pdbid}.voxels.pkl')
    if os.path.exists(voxelfile) and (not args.overwrite):
        print(f'Voxel file already exists: {voxelfile}.')
    else:
        lattice_coords = voxelise(args.pdbid, lig, prot, args.outdir, \
                args.pocket_cutoff, args.spacing, \
                args.trim, trim_cutoff, args.abs_include, \
                args.intermediate)

    # show contacting atoms, each item is a tuple
    # tuple:  bitvector, ligand atom indices and protein atom indices
    landmark_file = os.path.join(args.outdir, f'{args.pdbid}.landmarks.pkl')
    landmarks = {}
    if args.interactions:
        fp = plf.Fingerprint(interactions=args.interactions)
        ifp = fp.generate(lig, prot)
        int_landmarks = create_interaction_landmarks(ifp, lig)
        landmarks.update(int_landmarks)
    if args.heavyatom:
        atom_landmarks = create_heavy_atom_landmarks(lig)
        landmarks.update(atom_landmarks)
    if args.charge:
        charge_landmarks = create_partial_charge_landmarks(lig)
        landmarks.update(charge_landmarks)
    if args.intermediate:
        for interaction, coords in landmarks.items():
            to_xyz_file(coords, os.path.join(args.outdir, f'{args.pdbid}_{interaction}_landmarks.xyz'))
    with open(landmark_file, 'wb') as file:
        pickle.dump(landmarks, file)
    print("Succesfully voxelise structure and create landmarks:", args.pdbid)