import json

import numpy as np
import pandas as pd

laue_names = {
    'CI': 'Cubic I',
    'CII': 'Cubic II',
    'HI': 'Hexagonal I',
    'HII': 'Hexagonal II',
    'RI': 'Rhombohedral I',
    'RII': 'Rhombohedral II',
    'TI': 'Tetragonal I',
    'TII': 'Tetragonal II',
    'O': 'Orthorhombic',
    'Mb': 'Monoclinic Diad||x3',
    'Mc': 'Monoclinic Diad||x2',
    'N': 'Triclinic'
}

ulics = [np.array([1, 2, 3, 4, 5, 6]),
         np.array([-2, 1, 4, -3, 6, -5]),
         np.array([3, -5, -1, 6, 2, -4]),
         np.array([-4, -6, 5, 1, -3, 2]),
         np.array([5, 4, 6, -2, -1, -3]),
         np.array([-6, 3, -2, 5, -4, 1])]

ulics = [n / np.linalg.norm(n) for n in ulics]

ulics = {
    'CI': np.array([ulics[0]]),
    'CII': np.array([ulics[0]]),
    'HI': np.array([ulics[0],
                    ulics[2]]),
    'HII': np.array([ulics[0],
                     ulics[2]]),
    'RI': np.array([ulics[0],
                    ulics[2]]),
    'RII': np.array([ulics[0],
                     ulics[2]]),
    'TI': np.array([ulics[0],
                    ulics[2]]),
    'TII': np.array([ulics[0],
                     ulics[2]]),
    'O': np.array([ulics[0],
                   ulics[2],
                   ulics[4]]),
    'Mb': np.array([ulics[0],
                      ulics[1],
                      ulics[2],
                      ulics[3],
                      ulics[4]]),
    'Mc': np.array([ulics[0],
                    ulics[1],
                    ulics[2],
                    ulics[3],
                    ulics[4]]),
    'N': np.array([ulics[0],
                   ulics[1],
                   ulics[2],
                   ulics[3],
                   ulics[4],
                   ulics[5]]),
}


def load_olics(olics_file:str='../../make_OLICS/D_OLICS.json') -> dict:
    with open(olics_file, 'r') as file:
        olics_data = json.load(file)
    olics_dict = {laue: np.array(value['D-OLICS']) for laue, value in olics_data.items()}
    return olics_dict


olics = load_olics()


def string_to_tuple(stringin):
    # transform voigt index string to matrix tuple
    return tuple([int(i) - 1 for i in stringin])


def tuple_value_list_to_matrix(tuples_values_list):
    matrix = np.zeros((6, 6))
    for i in tuples_values_list:
        t, v = string_to_tuple(i[0]), i[1]
        symmt = tuple(reversed(t))
        # print i[0], t, symmt
        matrix[t] = v
        matrix[symmt] = v
    return matrix


def construct_elastic_symmetries(laue:str, filename:str='../../references/elastic_paper_table.csv',
                           returntype='matrix'):
    import os
    df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), filename)), dtype=str)
    indcomp = []
    comp_values = []
    components = df['N'].tolist()

    for line, comp in zip(df[laue].tolist(), components):
        if line == 'A*':
            indcomp.extend(['11', '12'])
        else:
            res = line.split('*')
            if len(res) == 2:
                indcomp.append(res[1])
            elif res[0] != '0':
                indcomp.append(res[0])

    independentconstants = list(set(indcomp))
    # Checked: works correctly

    symmetriesdict = {i: [] for i in independentconstants}
    # We keep the standard names for the constants a keys
    # for the dict
    # later we can standardize

    for line, comp in zip(df[laue].tolist(), components):
        # comp is the position in the elastic matrix
        # i.e. the position in the symmetry matrix
        # line is what the value at the position is
        # that might be e.g. another constant

        if line == 'A*':
            symmetriesdict['11'].append((comp, 0.5))
            symmetriesdict['12'].append((comp, -0.5))
        else:
            res = line.split('*')
            if len(res) == 2:
                symmetriesdict[res[1]].append((comp, int(res[0])))
            elif res[0] != '0':
                symmetriesdict[res[0]].append((comp, 1))

    symmetry_matrices = []
    symmetry_matrices_dict = {k: tuple_value_list_to_matrix(v) for k, v in symmetriesdict.items()}

    for kl in sorted(symmetry_matrices_dict.keys()):
        symmetry_matrices.append(symmetry_matrices_dict[kl])
    if returntype == 'matrix':
        return np.array(symmetry_matrices)
    else:
        return symmetry_matrices_dict




elastic_symmetries = {laue : construct_elastic_symmetries(laue) for laue in laue_names }
