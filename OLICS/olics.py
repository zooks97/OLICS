from multiprocessing import Pool

import numpy as np
import scipy.optimize as spopt

def get_laue_name(laue: str) -> str:
    """
    Look up the long name for a Laue group
    :param laue: Laue group symbol
    :returns: long name for Laue group
    """
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
    return laue_names[laue]


def get_ulics(laue: str) -> np.array:
    """
    Get the ULICS for a given Laue group
    :param laue: Laue group symbol
    :returns: ULICS array
    """
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
    
    return ulics[laue]


def get_min_num_strains(laue: str) -> int:
    """
    Get the minimum number of strain directions necessary for calculating
        elastic constants of a material with a given symmetry
    :param laue: Laue group symbol
    :returns: minimum number of strain directions
    """
    ulics = get_ulics(laue)
    min_num_strains = len(ulics)
    return min_num_strains


def gram_schmidt_rows(X: np.ndarray):
    """
    do a gram-schmidt orthonormalization on the rows of X using QR (take Q)
    :param X: 2-dimensional numpy array whose rows will be orthonormalized
    :returns: 2-dimensional orthonormal array with the same shape as X
    """
    Q, R = np.linalg.qr(X.transpose())
    return Q.transpose()


def get_strain_sets(num_sets: int, num_strains: int, num_dims: int=6, orthonormal: bool=False) -> np.ndarray:
    """
    generate random lagrangian strain sets
    :param num_sets: number of strain sets
    :param num_strains: number of strains in each strain set
    :param num_dims: number of strain components in each strain
    :param orthonormal: option to orthonormalize the strain sets
    :returns: (num_sets * num_strains) x num_dims matrix of strain sets
    """
    strain_sets = np.random.rand(num_sets * num_strains, num_dims) * 2 - 1
    strain_sets /= np.linalg.norm(strain_sets, axis=1)[:, np.newaxis]
    strain_sets = strain_sets.reshape(num_sets, num_strains, num_dims)
    if orthonormal and num_strains > 1:
        pool = Pool()
        orthstrain_sets = np.array(pool.map(gram_schmidt_rows, strain_sets))
        return orthstrain_sets
    else:
        return strain_sets


def get_elastic_vector(elastic_tensor: np.ndarray, symm_mat: np.ndarray) -> np.ndarray:
    """
    reduce an elastic tensor to an elastic vector using symmetries
    """
    elastic_vector = [[u for u in np.unique(elastic_tensor * symm) if u][0]
                for symm in symm_mat]
    return np.array(elastic_vector)



def norm_constraint_lower(x0: np.ndarray, strain_idx: int, num_strains: int, num_dims: int, tol: float) -> float:
    """
    Constrain the norm of the strain to be > 1 - tol
    :param x0: strains
    :param strain_idx: index of the strain to constrain
    :param num_strains: total number of strains
    :param num_dims: number of dimensions of the strain
    :param tol: constraint tolerance
    :returns: lower norm constraint
    """
    E = x0.reshape(num_strains, num_dims)
    e = E[strain_idx]
    constraints = []
    n = np.linalg.norm(e) - (1 - tol)
    constraints.append(n)
    return min(constraints)


def norm_constraint_upper(x0: np.ndarray, strain_idx: int, num_strains: int, num_dims: int, tol: float) -> float:
    """
    Constrain the norm of the strain to be < 1 + tol
    :param x0: strains
    :param strain_idx: index of the strain to constrain
    :param num_strains: total number of strains
    :param num_dims: number of dimensions of the strain
    :param tol: constraint tolerance
    :returns: upper norm constraint
    """
    E = x0.reshape(num_strains, num_dims)
    e = E[strain_idx]
    constraints = []
    n = (1 + tol) - np.linalg.norm(e)
    constraints.append(n)
    return min(constraints)


def norm_constraint_lower_jac(x0: np.ndarray, strain_idx: int, num_strains: int, num_dims: int) -> float:
    """
    Constrain the jacobian of the strain
    :param x0: strains
    :param strain_idx: index of the strain to constrain
    :param num_dims: number of dimeions of the strain
    """
    E = x0.reshape(num_strains, num_dims)
    jac = np.zeros((num_strains, num_dims))
    jac[strain_idx] = E[strain_idx]
    jac_norm = -jac / np.linalg.norm(jac)
    return jac_norm


def get_E_matrix(x0: np.ndarray, symm_mat: np.ndarray, num_strains: int, num_dims: int=6) -> np.ndarray:
    """
    We build up the experiment matrices E from the linear expressions
        a(gamma) = sum (sum M eta(gamma)) c with E(gamma) = (sum M eta(gamma))
    c is a vector with length N_a (independent elastic constants)
    E(gamma) is a 6 x N_a marix, obtained by multiplication of the symmetry
    matrix symm_mat(alpha, 6x6) for the lauegroup times the components of the
        deformation (6x1)
    In total we will stack num_strains E matrices together to form a
        6N_gamma x N_a matrix
    This is the important E matrix of the article.

    Careful: the way we stack together is a bit non-intuitive from numpy,
        but ok when done consistently, eg. when we stack the stresses
        (first derivatives along gamma)

    test :
    array([[[ 1,  2,  3,  4],
            [ 5,  6,  7,  8]],

           [[11, 12, 13, 14],
            [15, 16, 17, 18]]])

    np.vstack(test.T).T :
    array([[ 1,  5,  2,  6,  3,  7,  4,  8],
          [11, 15, 12, 16, 13, 17, 14, 18]])


    :param x0: strains careful with stacking and reshaping
    :param symm_mat: symmetry matrices for Laue group: (N_a x 6 x 6)
    :param num_strains: number of strains gamma (N_gamma in the article)
    :param num_dims: number of strain components in each strain
    :returns: information matrices E stacked into \bar{E}

    """

    Eta = x0.reshape(num_strains, num_dims)
    o_bar = symm_mat.copy()
    # this is transposed as it is of shape N_a x 6 x num_strains
    E_barT = np.matmul(o_bar, Eta.T)  
    # here we already have the right shape: 6num_strains x N_a
    E_bar_stacked = np.vstack(E_barT.T)  

    return E_bar_stacked


def doe_cost(x0: np.ndarray, symm_mat: np.ndarray, num_strains: int, num_dims: int=6, optimality: str='D', cij=None,
             normalize: bool=False) -> float:
    """
    1. Experiment matrices E were built in get_E_matrix
    2. Build the relevant ET E product
    3. Scale with cijguess
    4. Return the cost function according to optimality criterion

    :param x0: strains careful with stacking and reshaping
    :param symm_mat: symmetry matrices for Laue group: (N_a x 6 x 6)
    :param num_strains: number of strains gamma (N_gamma in the article)
    :param num_dims: number of strain components in each strain
    :param optimality: 'D', 'A', or 'E' optimailty
    :returns: cost associated with the selected optimality
    """
    
    if optimality not in ['D', 'A', 'E']:
        raise ValueError('{} is not a valid optimality (["D", "A", "E"] are valid)'.format(optimality))
    
    xuse = x0.copy()
    if normalize:
        D_strains = xuse.reshape(num_strains, num_dims)
        for n, strain in enumerate(D_strains):
            D_strains[n] /= np.linalg.norm(strain)

        xuse = D_strains.reshape(num_strains * num_dims)

    E = get_E_matrix(xuse, symm_mat, num_strains, num_dims=num_dims)

    EE = np.matmul(E.T, E)

    if cij is not None:
        o_bar = symm_mat.copy().astype(float)
        elastic_vector = get_elastic_vector(cij, o_bar)
        EE = EE * np.outer(elastic_vector, elastic_vector)

    # Do some linalg tricks with the inverse!!!

    if optimality == 'D':
        return 1. / np.max([1e-12, np.abs(np.linalg.det(EE))])

    elif optimality == 'A':
        try:
            EE_inv = np.linalg.inv(EE)

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                return 10 ** 9
            else:
                raise err

        return np.trace(EE_inv)

    elif optimality == 'E':
        try:
            EE_inv = np.linalg.inv(EE)

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                return 10 ** 9
            else:
                raise err

        return max(np.linalg.eigvals(EE_inv))


def optimize_locally(x0: np.ndarray, symm_mat: np.ndarray, num_strains: int, num_dims: int=6, cij_guess=None,
                     optimality: str='D', method: str='COBYLA', tol: float=0.01) -> np.ndarray:
    """
    Locally optimize the strain directions by minimizing a design of experiments
        cost function (i.e. D, A or E optimality)
    :param x0: strains
    :param symm_mat: symmetry matrices for Laue group
    :param num_strains: number of strains gamma
    :param num_dims: number of strain components in each strain
    :param cij_guess: guess for elastic constants matrix
    :param optimality: type of doe optimality, D, A, or E
    :param method: scipy minimization method, must accept constraints
    :param tol: tolerance on minimization
    """
    constraints = []
    for strain_n in range(num_strains):
        constraints += [{'type': 'ineq',
                         'fun': norm_constraint_lower,
                         'args': (strain_n, num_strains, num_dims, tol)},
                        {'type': 'ineq',
                         'fun': norm_constraint_upper,
                         'args': (strain_n, num_strains, num_dims, tol)}]

    args = (symm_mat, num_strains, num_dims, optimality, cij_guess)
    D_minimize = spopt.minimize(doe_cost, x0, args=args, method=method,
                                constraints=constraints, options = {'maxiter': 200, 'rhobeg': 0.001})
    D_strains = D_minimize.x.reshape(num_strains, num_dims)
    for n, strain in enumerate(D_strains):
        D_strains[n] /= np.linalg.norm(strain)

    return D_strains.reshape(num_strains * num_dims)


def optimize_basin_hopping(x0: np.ndarray, symm_mat: np.ndarray, num_strains: int, num_dims: int=6,
                           cij_guess=None, optimality: str='D', method: str='COBYLA',
                           tol: float=0.01, num_itermax: int=1000000, stepsize: float=0.2,
                           temp: float=20.0, ortho: bool=False, verbose: bool=False) -> np.nd_array:
    def take_normalized_step(x: np.ndarray, dims: int, self) -> np.nd_array:
        """
        We might be concerned with taking steps that are not isotropic if we
            rescale.
        This however seems not so problematic I would say as anyways the strain
            has his own kind of geometry and it is unclear how to interprete
            homogeneous sampling e.g. in strain space
        """
        x0 = x.copy()

        newx = x0 + (np.random.rand(len(x0)) - 0.5) * self.stepsize

        D_strains = newx.reshape(len(x) / dims, dims)
        for n, strain in enumerate(D_strains):
            D_strains[n] /= np.linalg.norm(strain)

        return D_strains.reshape(len(x))

    def take_normalized_step_ortho(x: np.nd_array, dims: int, self) -> np.ndarray:
        """
        We might be concerned with taking steps that are not isotropic if we
            rescale.
        This however seems not so problematic I would say as anyways the strain
            has his own kind of geometry and it is
            unclear how to interprete homogeneous sampling e.g. in strain space
        """
        x0 = x.copy()

        newx = x0 + (np.random.rand(len(x0)) - 0.5) * self.stepsize

        D_strains = newx.reshape(len(x) / dims, dims)
        for n, strain in enumerate(D_strains):
            D_strains[n] /= np.linalg.norm(strain)

        Xortho = gram_schmidt_rows(D_strains)
        return Xortho.reshape(len(x))

    # initialize random strain vectors
    for n, strain in enumerate(x0):
        x0[n] /= np.linalg.norm(strain)

    x0 = x0.reshape(num_strains * num_dims)

    take_normalized_step.func_defaults = (x0, num_dims, take_normalized_step)
    take_normalized_step.stepsize = stepsize

    take_normalized_step_ortho.func_defaults = (x0, num_dims,
        take_normalized_step_ortho)
    take_normalized_step_ortho.stepsize = stepsize

    constraints = []
    for strain_n in range(num_strains):
        constraints += [{'type': 'ineq',
                         'fun': norm_constraint_lower,
                         'args': (strain_n, num_strains, num_dims, tol)},
                        {'type': 'ineq',
                         'fun': norm_constraint_upper,
                         'args': (strain_n, num_strains, num_dims, tol)}]

    minimizer_kwargs = {'args': (symm_mat, num_strains, num_dims, optimality,
                                 cij_guess),
                        'method': method,
                        'constraints': constraints,
                        'options': {'maxiter': 200},
                        'tol': 0.0001}
    if ortho:
        resbasin = spopt.basinhopping(doe_cost, x0, niter=num_itermax, T=temp,
                                      stepsize=stepsize,
                                      minimizer_kwargs=minimizer_kwargs,
                                      take_step=take_normalized_step_ortho,
                                      accept_test=None, callback=None,
                                      interval=50, disp=verbose,
                                      niter_success=500)
    else:
        resbasin = spopt.basinhopping(doe_cost, x0, niter=num_itermax, T=temp,
                                      stepsize=stepsize,
                                      minimizer_kwargs=minimizer_kwargs,
                                      take_step=take_normalized_step,
                                      accept_test=None, callback=None,
                                      interval=50, disp=verbose,
                                      niter_success=500)

    D_strains = resbasin.x.reshape(num_strains, num_dims)
    for n, strain in enumerate(D_strains):
        D_strains[n] /= np.linalg.norm(strain)

    return D_strains.reshape(num_strains * num_dims)


def calc_cost(symmetry_matrix, num_strains, num_dims=6, optimality='D'):
    return lambda x0: doe_cost(x0, symmetry_matrix, num_strains, num_dims, optimality)


def basin_hopping(symmetry_matrix, num_strains, num_dims=6, cij_guess=None, optimality='D', method='SLSQP', tol=0.01,
                  num_itermax=1000, step_size=0.2, temp=10, orthogonal=True):
    return optimize_basin_hopping(x0, symmetry_matrix, num_strains, num_dims, cij_guess, optimality, method, tol, num_itermax, step_size, temp, orthogonal)