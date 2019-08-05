import numpy as np
import pandas as pd
import sys
import csv
import os

def voigt_index_to_matrix_index(voigt_index: str) -> tuple:
    """
    Transform a voigt index string (e.g. "11") to a matrix index tuple (e.g. (0,0))
    :param string: voigt index string (e.g. "11" for the first entry)
    :returns: equivalent matrix index (e.g. (0,0) for the first entry)
    """
    # TODO: implement some sanity checking assertions
    # only integers, within appropriate range, no zeros, etc.
    matrix_index = tuple([int(idx) - 1 for idx in voigt_index])
    return matrix_index


def voigt_indices_to_matrix(voigt_indices: list) -> np.ndarray:
    """
    Generate an elastic symmetry array from a list of symmetry-equivalent
        voigt indices
    :param voigt_indices: a list of voigt index strings (e.g. ["11", "22", "33"])
    :returns: an expanded symmetry matrix defined by voigt_indices
    """
    matrix = np.zeros((6,6))
    for idx in voigt_indices:
        matrix_index, voigt_index = voigt_index_to_matrix_index(idx[0]), idx[1]
        symmetry = tuple(reversed(matrix_index))
        matrix[matrix_index] = voigt_index
        matrix[symmetry] = voigt_index
    return matrix


def get_elastic_symmetries(laue_group: str,
                           df: pd.core.frame.DataFrame=pd.read_csv('./elastic_paper_table.csv', dtype=str)) -> np.ndarray:
    """
    Create elastic symmetry arrays based on a vector definition of the independent components
        of an elastic tensor and their prefactors
    :param laue_group: Laue group symmetry
    :param df: Symmetry data where columns are Laue group and rows are the value of the component
    :returns: 3D array where the first index returns a 2D 6x6 elastic tensor-like array
    """
    components = []
    component_values = []
    all_components = df['N'].tolist()  # all components present in no-symmetry case
    laue_components = df[laue_group].tolist()  # only components for requested laue symmetry group
    
    for laue_component, component in zip(laue_components, all_components):
        # A* = (11-12) / 2
        if laue_component == 'A*':
            components.extend(['11', '12'])
        else:
            split_component = laue_component.split('*')
            # the component is multiplied by some value
            if len(split_component) == 2:
                idx = split_component[1]
                components.append(idx)
            # the component is not multiplied by a value and is not 0
            elif split_component[0] != '0':
                idx = split_component[0]
                components.append(idx)
                
    independent_constants = list(set(components))
    symmetries = {idx: [] for idx in independent_constants}
    
    for laue_component, component in zip(laue_components, all_components):
        # get (index, prefactor) for present components and nothing for 0-value components 
        # A* = (11-12) / 2 always
        if laue_component == 'A*':
            symmetries['11'].append((component, 0.5))
            symmetries['12'].append((component, -0.5))
        else:
            split_component = laue_component.split('*')
            # the component is multiplied by some value (prefactor)
            if len(split_component) == 2:
                prefactor = int(split_component[0])
                idx = split_component[1]
                symmetries[idx].append((component, prefactor))
            # the component is not multiplied by a value and is not 0
            elif split_component[0] != '0':
                prefactor = 1
                idx = split_component[0]
                symmetries[idx].append((component, prefactor))
                
    symmetry_matrices = []
    symmetry_matrices_dict = {idx: voigt_indices_to_matrix(value) for idx, value in symmetries.items()}
    
    for component in sorted(symmetry_matrices_dict.keys()):
        symmetry_matrices.append(symmetry_matrices_dict[component])
        
    return np.array(symmetry_matrices)
