# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg
from pymatgen import Lattice, Structure
from spglib import standardize_cell

from elastic_data import elastic_symmetries, laue_names, olics, ulics

# Note about majority of matrices in numpy
# Because numpy is row major, stresses and
# strains are represented as row vectors
# Therefore, S = E @ C instead of the more
# common S = C @ E


class NotImplementedError(Exception):
    pass

# TODO: implement StiffnessTensor class for 6x6 elastic constants
class StiffnessTensor(pymatgen.core.tensors.Tensor):
    pass


# Properties Calculators

# Symmetry


# FIXME: check TI, TII, RI, RII, etc.
def spacegroup2laue(spacegroup_number: int) -> str:
    '''
    Return laue classification from space group number
    :param spacegroup_number: ITC space group number
    :return: laue classification
    '''

    if 1 <= spacegroup_number <= 2:  # Triclinic
        return 'N'
    elif 3 <= spacegroup_number <= 15:  # Monoclinic
        return 'Mb'
    elif 16 <= spacegroup_number <= 74:  # Orthorhombic
        return 'O'
    elif 75 <= spacegroup_number <= 88:  # Tetragonal II
        return 'TII'
    elif 89 <= spacegroup_number <= 142:  # Tetragonal I
        return 'TI'
    elif 143 <= spacegroup_number <= 148:  # Rhombohedral II
        return 'RII'
    elif 149 <= spacegroup_number <= 167:  # Rhombohedral I
        return 'RI'
    elif 168 <= spacegroup_number <= 176:  # Hexagonal II
        return 'HII'
    elif 177 <= spacegroup_number <= 194:  # Hexagonal I
        return 'HI'
    elif 195 <= spacegroup_number <= 206:  # Cubic II
        return 'CII'
    elif 207 <= spacegroup_number <= 230:  # Cubic I
        return 'CI'
    else:
        raise ValueError('Unknown space group number')


def get_number_of_elastic_constants(spacegroup_number: int) -> int:
    '''
    Return number of independent elastic constants for a specific space group
    number
    :param spacegroup_number:
    :return: Number of independent elastic constants
    '''
    if 1 <= spacegroup_number <= 2:  # Triclinic
        return 21
    elif 3 <= spacegroup_number <= 15:  # Monoclinic
        return 13
    elif 16 <= spacegroup_number <= 74:  # Orthorhombic
        return 9
    elif 75 <= spacegroup_number <= 88:  # Tetragonal II
        return 7
    elif 89 <= spacegroup_number <= 142:  # Tetragonal I
        return 6
    elif 143 <= spacegroup_number <= 148:  # Rhombohedral II
        return 7
    elif 149 <= spacegroup_number <= 167:  # Rhombohedral I
        return 6
    elif 168 <= spacegroup_number <= 176:  # Hexagonal II
        return 5
    elif 177 <= spacegroup_number <= 194:  # Hexagonal I
        return 5
    elif 195 <= spacegroup_number <= 206:  # Cubic II
        return 3
    elif 207 <= spacegroup_number <= 230:  # Cubic I
        return 3
    else:
        raise ValueError('Unknown space group number')


def get_min_number_of_strains(laue: str) -> int:
    '''
    Look up the minimum number of strain directions
        necessary to determine the system
    :param laue: laue classification string
    :return: int number of strains
    '''
    if laue in laue_names.keys():
        return len(ulics[laue])
    else:
        return ValueError('Invalid Laue classification')

# TODO: find expected return type
def get_elastic_symmetries(laue: str):
    '''
    Look up the elastic symmetries for the given
        laue classification
    :param laue: laue classification string
    :return: mx6x6 elastic symmetry matrix
    '''
    if laue in laue_names.keys():
        return elastic_symmetries[laue]
    else:
        return ValueError('Invalid Laue classification')


# Voigt

def get_voigt_bulk_modulus(elastic_tensor):
    '''
    voigt bulk modulus =
        ((c11 + c22 + c33) + 2(c12 + c23 + c31)) / 9
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: voigt bulk modulus
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        voigt_bulk_modulus = (
            et[0, 0] + et[1, 1] + et[2, 2] +
            2 * (et[0, 1] + et[0, 2] + et[1, 2])) / 9
        return voigt_bulk_modulus


def get_voigt_shear_modulus(elastic_tensor):
    '''
    voigt shear modulus =
        ((c11 + c22 + c33) - (c12 + c23 + c31) + 3(c44 + c55 + c66)) / 15
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: voigt shear modulus
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        voigt_shear_modulus = (
            (et[0, 0] + et[1, 1] + et[2, 2]) -
            (et[0, 1] + et[0, 2] + et[1, 2]) +
            3 * (et[3, 3] + et[4, 4] + et[5, 5])) / 15
        return voigt_shear_modulus


def get_voigt_youngs_modulus(elastic_tensor):
    '''
    young's modulus =
        (9 * bulk * shear) / (3 * bulk + shear)
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: voigt young's modulus
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        vbm = get_voigt_bulk_modulus(et)
        vsm = get_voigt_shear_modulus(et)
        voigt_youngs_modulus = (9 * vbm * vsm) / (3 * vbm + vsm)
        return voigt_youngs_modulus


def get_voigt_posison_ratio(elastic_tensor):
    '''
    isotropic poisson ratio =
        (1.5 * bulk - 2 * shear) / (6 * bulk + 2 * shear)
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: voigt isotropic poisson ratio
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        vbm = get_voigt_bulk_modulus(et)
        vsm = get_voigt_shear_modulus(et)
        voigt_poisson_ratio = (1.5 * vbm - vsm) / (3 * vbm + vsm)
        return voigt_poisson_ratio


# Reuss

def get_reuss_bulk_modulus(elastic_tensor):
    '''
    reuss bulk modulus =
        1 / ((s11 + s22 + s33) + 2 * (s12 + s23 + s31))
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: reuss bulk modulus
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        ct = elastic2compliance(et)
        reuss_bulk_modulus = 1 / \
            (ct[0, 0] + ct[1, 1] + ct[2, 2] +
             2 * (ct[0, 1] + ct[0, 2] + ct[1, 2]))
        return reuss_bulk_modulus


def get_reuss_shear_modulus(elastic_tensor):
    '''
    reuss shear modulus =
        15 / (4 * (s11 + s22 + s33) - 4 * \
              (s12 + s23 + s31) + 3 * (s44 + s55 + s66))
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: reuss shear modulus
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        ct = elastic2compliance(et)
        reuss_shear_modulus = 5 / (
            4 * (ct[0, 0] + ct[1, 1] + ct[2, 2]) -
            4 * (ct[0, 1] + ct[0, 2] + ct[1, 2]) +
            3 * (ct[3, 3] + ct[4, 4] + ct[5, 5]))
        return reuss_shear_modulus


def get_reuss_youngs_modulus(elastic_tensor):
    '''
    young's modulus =
        (9 * bulk * shear) / (3 * bulk + shear)
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: reuss young's modulus
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        ct = elastic2compliance(et)
        rbm = get_reuss_bulk_modulus(ct)
        rsm = get_reuss_shear_modulus(ct)
        reuss_youngs_modulus = (9 * rbm * rsm) / (3 * rbm + rsm)
        return reuss_youngs_modulus


def get_reuss_posison_ratio(elastic_tensor):
    '''
    isotropic poisson ratio =
        (1.5 * bulk - 2 * shear) / (6 * bulk + 2 * shear)
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: reuss isotropic poisson ratio
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        ct = elastic2compliance(et)
        rbm = get_reuss_bulk_modulus(ct)
        rsm = get_reuss_shear_modulus(ct)
        reuss_poisson_ratio = (1.5 * rbm - rsm) / (3 * rbm + rsm)
        return reuss_poisson_ratio


# Hill

def get_hill_bulk_modulus(elastic_tensor):
    '''
    hill bulk modulus =
        ((voigt bulk) + (hill bulk)) / 2
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: hill bulk modulus
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        vbm = get_voigt_bulk_modulus(et)
        rbm = get_reuss_bulk_modulus(et)
        hill_bulk_modulus = 0.5 * (vbm + rbm)
        return hill_bulk_modulus


def get_hill_shear_modulus(elastic_tensor):
    '''
    hill shear modulus =
        ((voigt shear modulus) + (reuss shear modulus)) / 2
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        vsm = get_voigt_shear_modulus(et)
        rsm = get_reuss_shear_modulus(et)
        hill_shear_modulus = 0.5 * (vsm + rsm)
        return hill_shear_modulus


def get_hill_youngs_modulus(elastic_tensor):
    '''
    young's modulus =
        (9 * bulk * shear) / (3 * bulk + shear)
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: hill young's modulus
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        hbm = get_hill_bulk_modulus(et)
        hsm = get_hill_shear_modulus(et)
        hill_youngs_modulus = (9 * hbm * hsm) / (3 * hbm + hsm)
        return hill_youngs_modulus


def get_hill_posison_ratio(elastic_tensor):
    '''
    isotropic poisson ratio =
        (1.5 * bulk - 2 * shear) / (6 * bulk + 2 * shear)
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: hill isotropic poisson ratio
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        hbm = get_hill_bulk_modulus(et)
        hsm = get_hill_shear_modulus(et)
        hill_poisson_ratio = (1.5 * hbm - hsm) / (3 * hbm + hsm)
        return hill_poisson_ratio


# Extra properties

def get_elastic_anisotropy(elastic_tensor):
    '''
    universal elastic anisotropy =
        (5 * (voigt shear) / (reuss shear) + (voigt bulk) / (reuss bulk) - 6)
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: universal elastic anisotropy
    '''
    et = np.array(elastic_tensor)
    if not et.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        rsm = get_reuss_shear_modulus(et)
        vsm = get_voigt_shear_modulus(et)
        rbm = get_reuss_bulk_modulus(et)
        vbm = get_voigt_bulk_modulus(et)
        universal_elastic_anisotropy = 5 * vsm / rsm + vbm / rbm - 6
        return universal_elastic_anisotropy


# Properties class

class ElasticProperties():
    def __init__(self, elastic_tensor):
        elastic_tensor = np.array(elastic_tensor)
        if not elastic_tensor.shape == (6, 6):
            raise ValueError('Elastic tensor matrix must be 6x6')
        else:
            self.elastic_tensor = elastic_tensor
            self.shear_stability_limit = 2  # GPa
            self.bulk_stability_limit = 2  # GPa

    @property
    def compliance_tensor(self):
        compliance_tensor = elastic2compliance(self.elastic_tensor)
        return compliance_tensor

    @property
    def voigt_bulk_modulus(self):
        voigt_bulk_modulus = get_voigt_bulk_modulus(self.elastic_tensor)
        return voigt_bulk_modulus

    @property
    def voigt_shear_modulus(self):
        voigt_shear_modulus = get_voigt_shear_modulus(self.elastic_tensor)
        return voigt_shear_modulus

    @property
    def voigt_youngs_modulus(self):
        voigt_youngs_modulus = get_voigt_youngs_modulus(self.elastic_tensor)
        return voigt_youngs_modulus

    @property
    def voigt_poisson_ratio(self):
        voigt_poisson_ratio = get_voigt_posison_ratio(self.elastic_tensor)
        return voigt_poisson_ratio

    @property
    def reuss_bulk_modulus(self):
        reuss_bulk_modulus = get_reuss_bulk_modulus(self.elastic_tensor)
        return reuss_bulk_modulus

    @property
    def reuss_shear_modulus(self):
        reuss_shear_modulus = get_reuss_shear_modulus(self.elastic_tensor)
        return reuss_shear_modulus

    @property
    def reuss_youngs_modulus(self):
        reuss_youngs_modulus = get_reuss_youngs_modulus(self.elastic_tensor)
        return reuss_youngs_modulus

    @property
    def reuss_poisson_ratio(self):
        reuss_poisson_ratio = get_reuss_posison_ratio(self.elastic_tensor)
        return reuss_poisson_ratio

    @property
    def hill_bulk_modulus(self):
        hill_bulk_modulus = get_hill_bulk_modulus(self.elastic_tensor)
        return hill_bulk_modulus

    @property
    def hill_shear_modulus(self):
        hill_shear_modulus = get_hill_shear_modulus(self.elastic_tensor)
        return hill_shear_modulus

    @property
    def hill_youngs_modulus(self):
        hill_youngs_modulus = get_hill_youngs_modulus(self.elastic_tensor)
        return hill_youngs_modulus

    @property
    def hill_poisson_ratio(self):
        hill_poisson_ratio = get_hill_posison_ratio(self.elastic_tensor)
        return hill_poisson_ratio

    @property
    def elastic_anisotropy(self):
        '''
        Universal elastic anisotropy index, isotropy = 0
        Source: Ranganathan SI, Ostoja-Starzewski M. Phys Rev Lett 2008:101.
        '''
        universal_elastic_anisotropy = get_elastic_anisotropy(
            self.elastic_tensor)
        return universal_elastic_anisotropy

    @property
    def eigenvalues(self):
        eigenvalues = np.linalg.eigvals(self.elastic_tensor)
        return eigenvalues

    @property
    def elastic_symmetries(self):
        raise NotImplementedError()

    @property
    def is_symmetric(self):
        return ((self.elastic_tensor.transpose() == self.elastic_tensor).all())

    @property
    def is_positive_definite(self):
        return (self.is_symmetric and (self.eigenvalues > 0).all())

    @property
    def is_bulk_stable(self):
        return (self.reuss_bulk_modulus > self.bulk_stability_limit)

    @property
    def is_shear_stable(self):
        return(self.reuss_shear_modulus > self.shear_stability_limit)

    def as_dict(self):
        elastic_dict = {
            'elastic_tensor': self.elastic_tensor,
            'voigt_bulk_modulus': self.voigt_bulk_modulus,
            'voigt_shear_modulus': self.voigt_shear_modulus,
            'voigt_youngs_modulus': self.voigt_youngs_modulus,
            'voigt_poisson_ratio': self.voigt_poisson_ratio,
            'reuss_bulk_modulus': self.reuss_bulk_modulus,
            'reuss_shear_modulus': self.reuss_shear_modulus,
            'reuss_youngs_modulus': self.reuss_youngs_modulus,
            'reuss_poisson_ratio': self.reuss_poisson_ratio,
            'hill_bulk_modulus': self.hill_bulk_modulus,
            'hill_shear_modulus': self.hill_shear_modulus,
            'hill_youngs_modulus': self.hill_youngs_modulus,
            'hill_poisson_ratio': self.hill_poisson_ratio,
            'elastic_anisotropy': self.elastic_anisotropy
        }
        return elastic_dict


def voigt2tensor_strain_voigt(voigt):
    '''
    :param voigt: 1x6 voigt notation strain vector (without / 2 terms)
    :return: 1x6 voigt notation tensor strain vector (with / 2 terms)
    '''
    voigt = np.array(voigt)
    if not voigt.shape == (6,):
        raise ValueError('Voigt vector must be 1x6')
    else:
        tensor_strain_voigt = np.array([voigt[0], voigt[1], voigt[2],
                                        voigt[3] / 2., voigt[4] / 2.,
                                        voigt[5] / 2.])
        return tensor_strain_voigt


def tensor_strain_voigt2voigt(tensor_strain_voigt):
    '''
    :param tensor_strain_voigt: 1x6 voigt notation tensor strain vector
        (with / 2 terms)
    :return: 1x6 voigt notation strain vector (without / 2 terms)
    '''
    tsv = np.array(tensor_strain_voigt)
    if not tsv.shape == (6,):
        raise ValueError('Voigt vector must be 1x6')

    else:
        voigt = np.array([tsv[0], tsv[1], tsv[2],
                          tsv[3] * 2., tsv[4] * 2., tsv[5] * 2.])
        return voigt


def voigt2tensor(voigt):
    '''
    :param voigt: 1x6 voigt notation vector
    :return: 3x3 symmetric tensor matrix
    '''
    voigt = np.array(voigt)
    if not voigt.shape == (6,):
        raise ValueError('Voigt vector must be 1x6')
    else:
        tensor = np.array([[voigt[0], voigt[5], voigt[4]],
                           [voigt[5], voigt[1], voigt[3]],
                           [voigt[4], voigt[3], voigt[2]]])
        return tensor


def tensor2voigt(tensor):
    '''
    :param tensor: 3x3 symmetric tensor matrix
    :return: 1x6 voigt notation vector
    '''
    tensor = np.array(tensor)
    if not np.isclose(tensor, tensor.T).all():  # check symmetry A' = A
        raise ValueError('Tensor matrix must be symmetric')
    elif not tensor.shape == (3, 3):
        raise ValueError('Tensor matrix must be 3x3')
    else:
        voigt = np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                          tensor[1, 2], tensor[0, 2], tensor[0, 1]])
        return voigt


def voigt2strain_tensor(voigt):
    '''
    :param voigt: 1x6 voigt notation strain vector (without / 2 terms)
    :return: 3x3 symmetric strain tensor matrix (with / 2 terms)
    '''
    voigt = np.array(voigt)
    if not voigt.shape == (6,):
        raise ValueError('Voigt vector must be 1x6')
    else:
        tensor_strain_voigt = voigt2tensor_strain_voigt(voigt)
        tensor = voigt2tensor(tensor_strain_voigt)
        return tensor


def strain_tensor2voigt(strain_tensor):
    '''
    :param strain_tensor: 3x3 symmetric strain tensor matrix (with / 2 terms)
    :return: 1x6 voigt notation strain vector (without / 2 terms)
    '''
    strain_tensor = np.array(strain_tensor)
    # check symmetry A' = A
    if not (strain_tensor.transpose() == strain_tensor).all():
        raise ValueError('Strain tensor matrix must be symmetric')
    elif not strain_tensor.shape == (3, 3):
        raise ValueError('Strain tensor matrix must be 3x3')
    else:
        # convert to voigt with / 2 terms
        tensor_strain_voigt = tensor2voigt(strain_tensor)
        # convert to voigt without / 2 terms
        voigt = tensor_strain_voigt2voigt(tensor_strain_voigt)
        return voigt


def elastic2compliance(elastic_tensor):
    '''
    :param elastic_tensor: 6x6 voigt notation elastic tensor matrix
    :return: 6x6 voigt notation compliance tensor matrix
    '''
    elastic_tensor = np.array(elastic_tensor)
    if not elastic_tensor.shape == (6, 6):
        raise ValueError('Elastic tensor matrix must be 6x6')
    else:
        return np.linalg.inv(elastic_tensor)


def normalize_lagrangian(lagrangian):
    '''
    :param lagrangian: 1x6 voigt notation lagrangian strain vector
    :return: normalized 1x6 voigt notation lagrangian strain vector
    '''
    lagrangian_tensor = voigt2tensor(
        lagrangian)  # voigt vector -> tensor matrix
    lagrangian_norm = np.linalg.norm(lagrangian_tensor)
    normalized_lagrangian = lagrangian / lagrangian_norm
    return normalized_lagrangian


def lagrangian2eulerian(lagrangian):
    '''
    eta = eps + 0.5 * eps^2
    :param lagrangian: 1x6 voigt notation lagrangian strain vector
    :return: 3x3 symmetric eulerian strain tensor matrix
    '''
    lagrangian = np.array(lagrangian)
    lagrangian_tensor = voigt2strain_tensor(lagrangian)
    FTF = 2 * lagrangian_tensor + np.eye(3)

    eulerian_tensor = scipy.linalg.sqrtm(FTF) - np.eye(3)

    if np.linalg.norm(eulerian_tensor - eulerian_tensor.T) >= 1e-10:
        raise ValueError('Eulerian tensor is not symmetric')

    return eulerian_tensor


def eulerian2lagrangian(eulerian):
    '''
    eta = eps + 0.5 * eps^2
    :param eulerian: 3x3 eulerian strain tensor matrix
    :return: 3x3 lagrangian strain tensor matrix
    '''
    lagrangian = eulerian + 0.5 * eulerian**2
    return lagrangian


def physical_stress2lagrangian_stress(physical_stress, deformation):
    '''
    Physical stress (e.g. from QE) -> Lagrangian stress
    :param physical_stress: 3x3 physical stress tensor matrix
    :param deformation: 3x3 deformation matrix
    :return: 3x3 lagrangian stress tensor matrix
    '''
    physical_stress = np.array(physical_stress)
    deformation = np.array(deformation)
    if not physical_stress.shape == (3, 3):
        raise ValueError('Physical stress must be 3x3')
    else:
        inv_deformation = np.linalg.inv(deformation)
        lagrangian_stress = np.linalg.det(deformation) * \
            np.dot(inv_deformation,
                   np.dot(physical_stress,
                          inv_deformation))
        return lagrangian_stress


def eulerian2deformation(eulerian_tensor):
    '''
    eulerian strain tensor matrix -> deformation matrix
    :param eulerian_tensor: 3x3 symmetric eulerian strain tensor matrix
    :return: 3x3 symmetric deformation matrix
    '''
    eulerian_tensor = np.array(eulerian_tensor)
    if not eulerian_tensor.shape == (3, 3):
        raise ValueError('Eulerian tensor matrix must be 3x3')
    else:
        deformation = np.identity(3) + eulerian_tensor
        return deformation


def lagrangian2deformation(lagrangian):
    '''
    lagrangian strain voigt vector -> deformation matrix
    :param lagrangian_tensor: 1x6 voigt notation lagrangian strain vector
    :return: 3x3 symmetric deformation matrix
    '''
    lagrangian = np.array(lagrangian)
    if not lagrangian.shape == (6,):
        raise ValueError('Lagrangian voigt vector must be 1x6')
    else:
        eulerian_tensor = lagrangian2eulerian(lagrangian)
        deformation = eulerian2deformation(eulerian_tensor)
        return deformation


def structure2cell(structure):
    '''
    Convert pymatgen structures to cell tuples for spglib
    with optional compositional anonymization
    Args:
        structure (pymatgen.core.Structure): pymatgen Structure object
        anonymize (bool): replace all species with hydrogen
    Returns:
        tuple: cell tuple (lattice, positions, numbers) for spglib
    '''
    lattice = structure.lattice.matrix
    positions = structure.frac_coords
    numbers = [site.specie.Z for site in structure.sites]
    return(lattice, positions, numbers)


def standardize_pymatgen(og_structure, to_primitive=False,
                         symprec=0.05):
    '''
    Standardize a pymatgen Structure object using spglib
    :param og_structure: original pymatgen Structure
    :param to_primitive: whether to primitivize
    :param symprec: symmetry tolerance
    '''
    og_cell = structure2cell(og_structure)
    standard_cell = standardize_cell(og_cell, to_primitive=to_primitive,
                                     symprec=symprec)
    if standard_cell:
        standard_structure = Structure(standard_cell[0],
                                       standard_cell[2],
                                       standard_cell[1])
        return standard_structure
    else:
        return og_structure


def symmetry_pymatgen(structure, symprec=0.01, angle_tolerance=5.0):
    '''
    Get symmetry info for a pymatgen Structure
    :param structure: pymatgen Structure
    :param symprec: symmetry tolerance
    :param angle_tolerance: angle tolerance
    :return: space group symbol, space group number, laue group symbol
    '''

    space_group_symbol, space_group_number = structure.get_space_group_info(
        symprec, angle_tolerance)
    laue_symbol = spacegroup2laue(space_group_number, space_group_symbol)

    return space_group_number, space_group_symbol, laue_symbol


def deform_ase(og_ase, lagrangian):
    '''
    Deform an ase atoms structure using a lagrangian strain
    :param og_ase: undeformed ASE Atoms
    :param lagrangian: 1x6 voigt notation lagrangian strain (without / 2 terms)
    :return: deformed ASE Atoms
    '''
    lagrangian_tensor = voigt2strain_tensor(lagrangian)
    deformation = lagrangian2deformation(lagrangian_tensor)

    og_cell = og_ase.cell
    new_cell = np.dot(og_cell, deformation)

    new_ase = og_ase.copy()
    new_ase.set_cell(new_cell, scale_atoms=True)  # not sure why to scale atoms

    return new_ase


def deform_pymatgen(og_structure, lagrangian):
    '''
    Deform a pymatgen Structure object using a lagrangian strain
    :param og_structure: undeformed Structure
    :param lagrangian: 1x6 voigt notation lagrangian strain (without / 2 terms)
    :return: deformed Structure
    '''
    deformation = lagrangian2deformation(lagrangian)

    og_cell = og_structure.lattice.matrix
    new_cell = np.dot(og_cell, deformation.T)

    new_structure = og_structure.copy()
    new_lattice = Lattice(new_cell)
    new_structure.modify_lattice(new_lattice)

    return new_structure


def get_elastic_tensor(lagrangian_stresses, lagrangian_strains,
                       elastic_symmetries):
    '''
    Calculate m elastic constants via linear least squares regression using n
        stress / strain pairs. elastic_symmetries should be a multidimensional
        mx6x6 array (basically a list of 6x6 matrices) where each 6x6
        sub-matrix has 1s in the positions of a uniqe elastic constant.
        e.g. c11 for cubic cases should be:
           [[1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]
        while the full elastic_symmetries for cubic would be 3x6x6 with one 6x6
        array being the example for c11.
    :param physical_stresses: n x 6 matrix of physical stresses
    :param lagrangian_stresses: n x 6 matrix of lagrangian stresses
    :param elastic_symmetries: m x 6 x 6 matrix of elastic constant symmetries
    '''
    # physical_stresses = np.array(physical_stresses)
    # lagrangian_strains = np.array(lagrangian_strains)
    # elastic_symmetries = np.array(elastic_symmetries)
    # # nx6 where n is the # of strains / stresses
    # # 2-dimensional array where stress voigt vectors are rows
    # S = []
    # for physical_stress, lagrangian_strain in zip(physical_stresses,
    #                                               lagrangian_strains):
    #     deformation = lagrangian2deformation(lagrangian_strain)
    #     lagrangian_stress = physical_stress2lagrangian_stress(
    #         physical_stress, deformation)
    #     lagrangian_stress_voigt = tensor2voigt(lagrangian_stress)
    #     S.append(lagrangian_stress_voigt)

    lagrangian_stresses = np.vstack(lagrangian_stresses)
    lagrangian_strains = np.vstack(lagrangian_strains)

    #!!!!!!!!!!!!!! This is so inconsistent!!! WTF
    if len(lagrangian_stresses[0].shape) == 2:
        S = np.array([tensor2voigt(stress) for stress in lagrangian_stresses])
    else:
        S = lagrangian_stresses
    # 1-dimensional array (vector) of stress voigt vectors appended
    # e.g. [[s11 s12 s13 s14 s15 s16]   ==>  [s11 .. s16 s21 .. s26]
    #       [s21 s22 s23 s24 s25 s26]]
    Sstack = np.hstack(S)  # 1x(n*6)

    # nx6 where n is the number of stresses / strains
    # 2-dimensional array where strain voigt vectors are rows
    E = lagrangian_strains

    # mx6x6 where m is the # of ind. elastic constants
    # 3-dimensional array where 2-dimensional arrays in the y-z plane are 6x6
    #     matrices representing elastic constant symmetries
    # in the x direction, different elastic constants are represented
    # e.g. cubic systems have 3 elastic constants, so this would be 3x6x6
    Obar = elastic_symmetries

    # symmetries scaled by strains
    # multiply each symmetry matrix (2-dimensional y-z matrix from above)
    #    by the set of strains applied to the system
    # this yields an nx6xm matrix (number of strains x 6 x number of
    #     symmetries)
    #    which will be useful for solving for elastic constants
    # Mbar = (Obar @ E.T).T
    Mbar = np.matmul(Obar, E.T).T

    # stack Mbar so that Mstack is a 2-dimensional array appropriate
    #    for linear least squares fitting
    # Mstack is (n*6)xm
    Mstack = np.vstack(Mbar)

    # fit Sstack = Mstack @ cbar
    # hooke's law has been recast so that
    #     S = CE
    # is equivalent to
    #     Sstack = (Mstack)cbar
    # where cbar is a mx1 vector of independent elastic constants in the order
    #     of how Mstack was stacked (order of symmetry matrices in
    #     Mbar / Mstack)
    cbar = np.linalg.lstsq(Mstack, Sstack, rcond=None)[0]

    # doing a tensordot between cbar and
    #     Obar (the 3-dimensional set of elastic constant symmetries)
    #     yields the 6x6 elastic tensor by multiplying each 6x6 symmetry matrix
    #     by the magnitude of its corresponding elastic constant and summing
    #     over the scaled symmetry matrices
    elastic_tensor = np.tensordot(cbar, Obar, axes=1)

    return elastic_tensor
