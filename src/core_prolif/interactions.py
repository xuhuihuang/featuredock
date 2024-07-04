import warnings
from abc import ABC, ABCMeta, abstractmethod
from itertools import product
from math import radians
from math import pi
import numpy as np
from rdkit import Geometry
from rdkit.Chem import MolFromSmarts
from rdkit.Geometry import Point3D


_INTERACTIONS = {}
_90_deg_to_rad = pi/2

def get_centroid(coordinates):
    """Centroid for an array of XYZ coordinates"""
    return np.mean(coordinates, axis=0)


def get_ring_normal_vector(centroid, coordinates):
    """Returns a vector that is normal to the ring plane"""
    # A & B are two edges of the ring
    a = Point3D(*coordinates[0])
    b = Point3D(*coordinates[1])
    # vectors between centroid and these edges
    ca = centroid.DirectionVector(a)
    cb = centroid.DirectionVector(b)
    # cross product between these two vectors
    normal = ca.CrossProduct(cb)
    # cb.CrossProduct(ca) is the normal vector in the opposite direction
    return normal


def angle_between_limits(angle, min_angle, max_angle, ring=False):
    """Checks if an angle value is between min and max angles in radian.
    If the angle to check involves a ring, include the angle that would be
    obtained if we had used the other normal vector (same axis but opposite
    direction)
    """
    if ring and (angle > _90_deg_to_rad):
        mirror_angle = _90_deg_to_rad - (angle % _90_deg_to_rad)
        return (min_angle <= angle <= max_angle) or (
                min_angle <= mirror_angle <= max_angle)
    return (min_angle <= angle <= max_angle)


class _InteractionMeta(ABCMeta):
    """Metaclass to register interactions automatically"""
    def __init__(cls, name, bases, classdict):
        type.__init__(cls, name, bases, classdict)
        if name in _INTERACTIONS.keys():
            warnings.warn(f"The {name!r} interaction has been superseded by a "
                          f"new class with id {id(cls):#x}")
        _INTERACTIONS[name] = cls


class Interaction(ABC, metaclass=_InteractionMeta):
    """Abstract class for interactions
    All interaction classes must inherit this class and define a
    :meth:`~detect` method
    """
    @abstractmethod
    def detect(self, **kwargs):
        pass


class _Distance(Interaction):
    """Generic class for distance-based interactions
    Parameters
    ----------
    lig_pattern : str
        SMARTS pattern for atoms in ligand residues
    prot_pattern : str
        SMARTS pattern for atoms in protein residues
    distance : float
        Cutoff distance, measured between the first atom of each pattern
    """
    def __init__(self, lig_pattern, prot_pattern, distance):
        self.lig_pattern = MolFromSmarts(lig_pattern)
        self.prot_pattern = MolFromSmarts(prot_pattern)
        self.distance = distance

    def detect(self, lig_res, prot_res):
        lig_matches = lig_res.GetSubstructMatches(self.lig_pattern)
        prot_matches = prot_res.GetSubstructMatches(self.prot_pattern)
        ilig = []
        ires = []
        if lig_matches and prot_matches:
            for lig_match, prot_match in product(lig_matches,
                                                           prot_matches):
                alig = Geometry.Point3D(*lig_res.xyz[lig_match[0]])
                aprot = Geometry.Point3D(*prot_res.xyz[prot_match[0]])
                if alig.Distance(aprot) <= self.distance:
                    ilig.append(lig_match[0])
                    ires.append(prot_match[0])
        if len(ilig) == 0:
            return False, None, None
        else:
            return True, tuple(ilig), tuple(ires)


class Hydrophobic(_Distance):
    """Hydrophobic interaction
    Parameters
    ----------
    hydrophobic : str
        SMARTS query for hydrophobic atoms
    distance : float
        Cutoff distance for the interaction
    """
    def __init__(self,
                 hydrophobic="[#6,#16,F,Cl,Br,I,At;+0]",
                 distance=4.5):
        super().__init__(hydrophobic, hydrophobic, distance)


class _BaseHBond(Interaction):
    """Base class for Hydrogen bond interactions
    Parameters
    ----------
    donor : str
        SMARTS for ``[Donor]-[Hydrogen]``
    acceptor : str
        SMARTS for ``[Acceptor]``
    distance : float
        Cutoff distance between the donor and acceptor atoms
    angles : tuple
        Min and max values for the ``[Donor]-[Hydrogen]...[Acceptor]`` angle
    """
    def __init__(self,
                 donor="[#7,#8,#16][H]",
                 acceptor="[N,O,F,-{1-};!+{1-}]",
                 distance=3.5,
                 angles=(130, 180)):
        self.donor = MolFromSmarts(donor)
        self.acceptor = MolFromSmarts(acceptor)
        self.distance = distance
        self.angles = tuple(radians(i) for i in angles)

    def detect(self, acceptor, donor):
        acceptor_matches = acceptor.GetSubstructMatches(self.acceptor)
        donor_matches = donor.GetSubstructMatches(self.donor)
        ilig = []
        ires = []
        if acceptor_matches and donor_matches:
            for donor_match, acceptor_match in product(donor_matches,
                                                       acceptor_matches):
                # D-H ... A
                d = Geometry.Point3D(*donor.xyz[donor_match[0]])
                h = Geometry.Point3D(*donor.xyz[donor_match[1]])
                a = Geometry.Point3D(*acceptor.xyz[acceptor_match[0]])
                if d.Distance(a) <= self.distance:
                    ilig.append(acceptor_match[0])
                    # ires.extend(donor_match)
                    ires.append(donor_match[0])
        if len(ilig) == 0:
            return False, None, None
        else:
            return True, tuple(ilig), tuple(ires)


class HBDonor(_BaseHBond):
    """Hbond interaction between a ligand (donor) and a residue (acceptor)"""
    def detect(self, ligand, residue):
        bit, ires, ilig = super().detect(residue, ligand)
        return bit, ilig, ires


class HBAcceptor(_BaseHBond):
    """Hbond interaction between a ligand (acceptor) and a residue (donor)"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class _BaseCationPi(Interaction):
    """Base class for cation-pi interactions
    Parameters
    ----------
    cation : str
        SMARTS for cation
    pi_ring : tuple
        SMARTS for aromatic rings (5 and 6 membered rings only)
    distance : float
        Cutoff distance between the centroid and the cation
    angles : tuple
        Min and max values for the angle between the vector normal to the ring
        plane and the vector going from the centroid to the cation
    """
    def __init__(self,
                 cation="[+{1-}]",
                 pi_ring=("a1:a:a:a:a:a:1", "a1:a:a:a:a:1"),
                 distance=4.5,
                 angles=(0, 30)):
        self.cation = MolFromSmarts(cation)
        self.pi_ring = [MolFromSmarts(s) for s in pi_ring]
        self.distance = distance
        self.angles = tuple(radians(i) for i in angles)

    def detect(self, cation, pi):
        cation_matches = cation.GetSubstructMatches(self.cation)
        ilig = []
        ires = []
        for pi_ring in self.pi_ring:
            pi_matches = pi.GetSubstructMatches(pi_ring)
            if not (cation_matches and pi_matches):
                continue
            for cation_match, pi_match in product(cation_matches, pi_matches):
                cat = Geometry.Point3D(*cation.xyz[cation_match[0]])
                # get coordinates of atoms matching pi-system
                pi_coords = pi.xyz[list(pi_match)]
                # centroid of pi-system as 3d point
                centroid = Geometry.Point3D(*get_centroid(pi_coords))
                # distance between cation and centroid
                if cat.Distance(centroid) > self.distance:
                    continue
                # vector normal to ring plane
                normal = get_ring_normal_vector(centroid, pi_coords)
                # vector between the centroid and the charge
                centroid_cation = centroid.DirectionVector(cat)
                # compute angle between normal to ring plane and
                # centroid-cation
                angle = normal.AngleTo(centroid_cation)
                if angle_between_limits(angle, *self.angles, ring=True):
                    ilig.append(cation_match[0])
                    ires.extend(pi_match)
        if len(ilig) == 0:
            return False, None, None
        else:
            return True, tuple(ilig), tuple(ires)


class PiCation(_BaseCationPi):
    """Cation-Pi interaction between a ligand (aromatic ring) and a residue
    (cation)"""
    def detect(self, ligand, residue):
        bit, ires, ilig = super().detect(residue, ligand)
        return bit, ilig, ires


class CationPi(_BaseCationPi):
    """Cation-Pi interaction between a ligand (cation) and a residue
    (aromatic ring)"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class PiStacking(Interaction):
    """Pi-Stacking interaction between a ligand and a residue
    Parameters
    ----------
    centroid_distance : float
        Cutoff distance between each rings centroid
    shortest_distance : float
        Shortest distance allowed between the closest atoms of both rings
    plane_angles : tuple
        Min and max values for the angle between the ring planes
    pi_ring : list
        List of SMARTS for aromatic rings
    """
    def __init__(self,
                 centroid_distance=6.0,
                 shortest_distance=3.8,
                 plane_angles=(0, 90),
                 pi_ring=("a1:a:a:a:a:a:1", "a1:a:a:a:a:1")):
        self.pi_ring = [MolFromSmarts(s) for s in pi_ring]
        self.centroid_distance = centroid_distance
        self.shortest_distance = shortest_distance**2
        self.plane_angles = tuple(radians(i) for i in plane_angles)

    def detect(self, ligand, residue):
        ilig = []
        ires = []
        for pi_rings in product(self.pi_ring, repeat=2):
            res_matches = residue.GetSubstructMatches(pi_rings[0])
            lig_matches = ligand.GetSubstructMatches(pi_rings[1])
            if not (lig_matches and res_matches):
                continue
            for lig_match, res_match in product(lig_matches, res_matches):
                lig_pi_coords = ligand.xyz[list(lig_match)]
                lig_centroid = Geometry.Point3D(*get_centroid(lig_pi_coords))
                res_pi_coords = residue.xyz[list(res_match)]
                res_centroid = Geometry.Point3D(*get_centroid(res_pi_coords))
                cdist = lig_centroid.Distance(res_centroid)
                if cdist > self.centroid_distance:
                    continue
                squared_dist_matrix = np.add.outer(
                    (lig_pi_coords**2).sum(axis=-1),
                    (res_pi_coords**2).sum(axis=-1)
                ) - 2*np.dot(lig_pi_coords, res_pi_coords.T)
                shortest_dist = squared_dist_matrix.min().min()
                if shortest_dist > self.shortest_distance:
                    continue
                # ligand
                lig_normal = get_ring_normal_vector(lig_centroid,
                                                    lig_pi_coords)
                # residue
                res_normal = get_ring_normal_vector(res_centroid,
                                                    res_pi_coords)
                # angle between planes
                plane_angle = lig_normal.AngleTo(res_normal)
                if angle_between_limits(plane_angle, *self.plane_angles,
                                        ring=True):
                    ilig.extend(lig_match)
                    ires.extend(res_match)
        if len(ilig) == 0:
            return False, None, None
        else:
            return True, tuple(ilig), tuple(ires)


class _BaseIonic(_Distance):
    """Base class for ionic interactions"""
    def __init__(self,
                 cation="[+{1-}]",
                 anion="[-{1-}]",
                 distance=4.5):
        super().__init__(cation, anion, distance)


class Cationic(_BaseIonic):
    """Ionic interaction between a ligand (cation) and a residue (anion)"""
    def detect(self, ligand, residue):
        return super().detect(ligand, residue)


class Anionic(_BaseIonic):
    """Ionic interaction between a ligand (anion) and a residue (cation)"""
    def detect(self, ligand, residue):
        bit, ires, ilig = super().detect(residue, ligand)
        return bit, ilig, ires
