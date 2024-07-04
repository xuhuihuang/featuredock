import copy
import re
from collections import UserDict
from typing import List, Optional
from operator import attrgetter
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit.Chem import FragmentOnBonds, GetMolFrags, SplitMolByPDBResidues
from scipy.spatial import cKDTree

_RE_RESID = re.compile(r'([A-Z]{,3})?(\d*)\.?(\w)?')
NoneType = type(None)

class BaseRDKitMol(Chem.Mol):
    """Base molecular class that behaves like an RDKit :class:`~rdkit.Chem.rdchem.Mol`
    with extra attributes (see below).
    The sole purpose of this class is to define the common API between the
    :class:`~prolif.molecule.Molecule` and :class:`~prolif.residue.Residue` classes.
    This class should not be instantiated by users.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A molecule (protein, ligand, or residue) with a single conformer

    Attributes
    ----------
    centroid : numpy.ndarray
        XYZ coordinates of the centroid of the molecule
    xyz : numpy.ndarray
        XYZ coordinates of all atoms in the molecule
    """
    @property
    def centroid(self):
        return ComputeCentroid(self.GetConformer())

    @property
    def xyz(self):
        return self.GetConformer().GetPositions()

class ResidueId:
    """A unique residue identifier

    Parameters
    ----------
    name : str
        3-letter residue name
    number : int
        residue number
    chain : str or None, optionnal
        1-letter protein chain
    """
    def __init__(self,
                 name: str = "UNK",
                 number: int = 0,
                 chain: Optional[str] = None):
        self.name = name or "UNK"
        self.number = number or 0
        self.chain = chain or None

    def __repr__(self):
        return f"ResidueId({self.name}, {self.number}, {self.chain})"

    def __str__(self):
        resid = f"{self.name}{self.number}"
        if self.chain:
            resid += f".{self.chain}"
        return resid

    def __hash__(self):
        return hash((self.name, self.number, self.chain))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __lt__(self, other):
        return (self.chain, self.number) < (other.chain, other.number)

    @classmethod
    def from_atom(cls, atom):
        """Creates a ResidueId from an RDKit atom

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            An atom that contains an RDKit :class:`~rdkit.Chem.rdchem.AtomMonomerInfo`
        """
        mi = atom.GetMonomerInfo()
        if mi:
            name = mi.GetResidueName()
            number = mi.GetResidueNumber()
            chain = mi.GetChainId()
            return cls(name, number, chain)
        return cls()

    @classmethod
    def from_string(cls, resid_str):
        """Creates a ResidueId from a string

        Parameters
        ----------
        resid_str : str
            A string in the format ``<3-letter code><residue number>.<chain>``
            All arguments are optionnal, and the dot should be present only if
            the chain identifier is also present

        Examples
        --------

        +-----------+----------------------------------+
        | string    | Corresponding ResidueId          |
        +===========+==================================+
        | "ALA10.A" | ``ResidueId("ALA", 10, "A")``    |
        +-----------+----------------------------------+
        | "GLU33"   | ``ResidueId("GLU", 33, None)``   |
        +-----------+----------------------------------+
        | "LYS.B"   | ``ResidueId("LYS", 0, "B")``     |
        +-----------+----------------------------------+
        | "ARG"     | ``ResidueId("ARG", 0, None)``    |
        +-----------+----------------------------------+
        | "5.C"     | ``ResidueId("UNK", 5, "C")``     |
        +-----------+----------------------------------+
        | "42"      | ``ResidueId("UNK", 42, None)``   |
        +-----------+----------------------------------+
        | ".D"      | ``ResidueId("UNK", 0, "D")``     |
        +-----------+----------------------------------+
        | ""        | ``ResidueId("UNK", 0, None)``    |
        +-----------+----------------------------------+

        """
        matches = _RE_RESID.search(resid_str)
        name, number, chain = matches.groups()
        number = int(number) if number else 0
        return cls(name, number, chain)


class Residue(BaseRDKitMol):
    """A class for residues as RDKit molecules

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The residue as an RDKit molecule

    Attributes
    ----------
    resid : prolif.residue.ResidueId
        The residue identifier

    Notes
    -----
    The name of the residue can be converted to a string by using
    ``str(Residue)``
    """
    def __init__(self, mol):
        super().__init__(mol)
        self.resid = ResidueId.from_atom(self.GetAtomWithIdx(0))

    def __repr__(self): # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} {self.resid} at {id(self):#x}>"

    def __str__(self):
        return str(self.resid)


class ResidueGroup(UserDict):
    """A container to store and retrieve Residue instances easily

    Parameters
    ----------
    residues : list
        A list of :class:`~prolif.residue.Residue`

    Attributes
    ----------
    n_residues : int
        Number of residues in the ResidueGroup

    Notes
    -----
    Residues in the group can be accessed by :class:`ResidueId`, string, or
    index. See the :class:`~prolif.molecule.Molecule` class for an example.
    You can also use the :meth:`~prolif.residue.ResidueGroup.select` method to
    access a subset of a ResidueGroup.
    """
    def __init__(self, residues: List[Residue]):
        self._residues = np.asarray(residues, dtype=object)
        resinfo = [(r.resid.name, r.resid.number, r.resid.chain)
                   for r in self._residues]
        try:
            name, number, chain = zip(*resinfo)
        except ValueError:
            self.name = np.array([], dtype=object)
            self.number = np.array([], dtype=np.uint8)
            self.chain = np.array([], dtype=object)
        else:
            self.name = np.asarray(name, dtype=object)
            self.number = np.asarray(number, dtype=np.uint16)
            self.chain = np.asarray(chain, dtype=object)
        super().__init__([(r.resid, r) for r in self._residues])

    def __getitem__(self, key):
        # bool is a subclass of int but shouldn't be used here
        if isinstance(key, bool):
            raise KeyError("Expected a ResidueId, int, or str, "
                           f"got {type(key).__name__!r} instead")
        if isinstance(key, int):
            return self._residues[key]
        elif isinstance(key, str):
            key = ResidueId.from_string(key)
            return self.data[key]
        elif isinstance(key, ResidueId):
            return self.data[key]
        raise KeyError("Expected a ResidueId, int, or str, "
                       f"got {type(key).__name__!r} instead")

    def select(self, mask):
        """Locate a subset of a ResidueGroup based on a boolean mask

        Parameters
        ----------
        mask : numpy.ndarray
            A 1D array of ``dtype=bool`` with the same length as the number of
            residues in the ResidueGroup. The mask should be constructed by
            using conditions on the "name", "number", and "chain" residue
            attributes as defined in the :class:`~prolif.residue.ResidueId`
            class

        Returns
        -------
        rg : prolif.residue.ResidueGroup
            A subset of the original ResidueGroup

        Examples
        --------
        ::

            >>> rg
            <prolif.residue.ResidueGroup with 200 residues at 0x7f9a68719ac0>
            >>> rg.select(rg.chain == "A")
            <prolif.residue.ResidueGroup with 42 residues at 0x7fe3fdb86ca0>
            >>> rg.select((10 <= rg.number) & (rg.number < 30))
            <prolif.residue.ResidueGroup with 20 residues at 0x7f5f3c69aaf0>
            >>> rg.select((rg.chain == "B") & (np.isin(rg.name, ["ASP", "GLU"])))
            <prolif.residue.ResidueGroup with 3 residues at 0x7f5f3c510c70>

        As seen in these examples, you can combine masks with different
        operators, similarly to numpy boolean indexing or pandas
        :meth:`~pandas.DataFrame.loc` method

            * AND --> ``&``
            * OR --> ``|``
            * XOR --> ``^``
            * NOT --> ``~``

        """
        return ResidueGroup(self._residues[mask])

    def __repr__(self): # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        return f"<{name} with {self.n_residues} residues at {id(self):#x}>"

    @property
    def n_residues(self):
        return len(self)


class Molecule(BaseRDKitMol):
    """Main molecule class that behaves like an RDKit :class:`~rdkit.Chem.rdchem.Mol`
    with extra attributes (see examples below). The main purpose of this class
    is to access residues as fragments of the molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        A ligand or protein with a single conformer

    Attributes
    ----------
    residues : prolif.residue.ResidueGroup
        A dictionnary storing one/many :class:`~prolif.residue.Residue` indexed
        by :class:`~prolif.residue.ResidueId`. The residue list is sorted.
    n_residues : int
        Number of residues

    Examples
    --------

    .. ipython:: python
        :okwarning:

        import MDAnalysis as mda
        import prolif
        u = mda.Universe(prolif.datafiles.TOP, prolif.datafiles.TRAJ)
        mol = u.select_atoms("protein").convert_to("RDKIT")
        mol = prolif.Molecule(mol)
        mol

    You can also create a Molecule directly from a
    :class:`~MDAnalysis.core.universe.Universe`:

    .. ipython:: python
        :okwarning:

        mol = prolif.Molecule.from_mda(u, "protein")
        mol


    Notes
    -----
    Residues can be accessed easily in different ways:

    .. ipython:: python

        mol["TYR38.A"] # by resid string (residue name + number + chain)
        mol[42] # by index (from 0 to n_residues-1)
        mol[prolif.ResidueId("TYR", 38, "A")] # by ResidueId

    See :mod:`prolif.residue` for more information on residues
    """
    def __init__(self, mol):
        super().__init__(mol)
        # set mapping of atoms
        for atom in self.GetAtoms():
            atom.SetUnsignedProp("mapindex", atom.GetIdx())
        # split in residues
        residues = split_mol_by_residues(self)
        residues = [Residue(mol) for mol in residues]
        residues.sort(key=attrgetter("resid"))
        self.residues = ResidueGroup(residues)
    
    @classmethod
    def from_rdkit(cls, mol, resname="UNL", resnumber=1, chain=""):
        """Creates a Molecule from an RDKit molecule

        While directly instantiating a molecule with ``prolif.Molecule(mol)``
        would also work, this method insures that every atom is linked to an
        AtomPDBResidueInfo which is required by ProLIF

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            The input RDKit molecule
        resname : str
            The default residue name that is used if none was found
        resnumber : int
            The default residue number that is used if none was found
        chain : str
            The default chain Id that is used if none was found

        Notes
        -----
        This method only checks for an existing AtomPDBResidueInfo in the first
        atom. If none was found, it will patch all atoms with the one created
        from the method's arguments (resname, resnumber, chain).
        """
        if mol.GetAtomWithIdx(0).GetMonomerInfo():
            return cls(mol)
        mol = copy.deepcopy(mol)
        for atom in mol.GetAtoms():
            mi = Chem.AtomPDBResidueInfo(f" {atom.GetSymbol():<3.3}",
                                         residueName=resname,
                                         residueNumber=resnumber,
                                         chainId=chain)
            atom.SetMonomerInfo(mi)
        return cls(mol)

    def __iter__(self):
        for residue in self.residues.values():
            yield residue

    def __getitem__(self, key):
        return self.residues[key]

    def __repr__(self): # pragma: no cover
        name = ".".join([self.__class__.__module__, self.__class__.__name__])
        params = f"{self.n_residues} residues and {self.GetNumAtoms()} atoms"
        return f"<{name} with {params} at {id(self):#x}>"

    @property
    def n_residues(self):
        return len(self.residues)


def get_residues_near_ligand(lig, prot, cutoff=6.0):
    """Detects residues close to a reference ligand

    Parameters
    ----------
    lig : prolif.molecule.Molecule
        Select residues that are near this ligand
    prot : prolif.molecule.Molecule
        Protein containing the residues
    cutoff : float
        If any interatomic distance between the ligand reference points and a
        residue is below or equal to this cutoff, the residue will be selected

    Returns
    -------
    residues : list
        A list of unique :class:`~prolif.residue.ResidueId` that are close to
        the ligand
    """
    tree = cKDTree(prot.xyz)
    ix = tree.query_ball_point(lig.xyz, cutoff)
    ix = set([i for lst in ix for i in lst])
    resids = [ResidueId.from_atom(prot.GetAtomWithIdx(i)) for i in ix]
    return list(set(resids))


def is_peptide_bond(bond, resids):
    """Checks if a bond is a peptide bond based on the ResidueId of the atoms
    on each part of the bond. Also works for disulfide bridges or any bond that
    links two residues in biopolymers.

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        The bond to check
    resids : dict
        A dictionnary of ResidueId indexed by atom index
    """
    return resids[bond.GetBeginAtomIdx()] != resids[bond.GetEndAtomIdx()]


def split_mol_by_residues(mol):
    """Splits a molecule in multiple fragments based on residues

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The molecule to fragment

    Returns
    -------
    residues : list
        A list of :class:`rdkit.Chem.rdchem.Mol`

    Notes
    -----
    Code adapted from Maciek Wójcikowski on the RDKit discussion list
    """
    residues = []
    for res in SplitMolByPDBResidues(mol).values():
        for frag in GetMolFrags(res, asMols=True, sanitizeFrags=False):
            # count number of unique residues in the fragment
            resids = {a.GetIdx(): ResidueId.from_atom(a)
                      for a in frag.GetAtoms()}
            if len(set(resids.values())) > 1:
                # split on peptide bonds
                bonds = [b.GetIdx() for b in frag.GetBonds()
                         if is_peptide_bond(b, resids)]
                mols = FragmentOnBonds(frag, bonds, addDummies=False)
                mols = GetMolFrags(mols, asMols=True, sanitizeFrags=False)
                residues.extend(mols)
            else:
                residues.append(frag)
    return residues
