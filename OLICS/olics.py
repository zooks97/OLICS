from multiprocessing import Pool

import numpy as np
import scipy.optimize as spopt
from pathos.multiprocessing import ProcessingPool

from .elastic import get_min_num_strains, get_elastic_symmetries


def gram_schmidt_rows(X: np.ndarray):
    """
    do a gram-schmidt orthonormalization on the rows of X using QR (take Q)
    :param X: 2-dimensional numpy array whose rows will be orthonormalized
    :returns: 2-dimensional orthonormal array with the same shape as X
    """
    Q, R = np.linalg.qr(X.transpose(), mode='complete')
    Q = Q.transpose()
    # If more than 6 strains are requested, the system will be over-determined
    # The first six strains will be orthonormal, but the final one will be free
    #     and simply normalized without any orthogonality constraint
    #     (this seems fine for the application)
    if X.shape[0] > X.shape[1]:
        add_row = X[-1]
        add_row /= np.linalg.norm(add_row)
        Q = np.vstack([Q, add_row])
    return Q


def get_strain_sets(num_sets: int, num_strains: int, num_dims: int=6,
                    orthonormal: bool=False) -> np.ndarray:
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



def norm_constraint_lower(x0: np.ndarray, strain_idx: int, num_strains: int,
                          num_dims: int, tol: float) -> float:
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


def norm_constraint_upper(x0: np.ndarray, strain_idx: int, num_strains: int,
                          num_dims: int, tol: float) -> float:
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


def norm_constraint_lower_jac(x0: np.ndarray, strain_idx: int, num_strains: int,
                              num_dims: int) -> float:
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


def get_E_matrix(x0: np.ndarray, symm_mat: np.ndarray, num_strains: int,
                 num_dims: int=6) -> np.ndarray:
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


def doe_cost(x0: np.ndarray, symm_mat: np.ndarray, num_strains: int, num_dims: int=6,
             optimality: str='D', cij=None, normalize: bool=False) -> float:
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
        raise ValueError('{} is not a valid optimality '
                         '(["D", "A", "E"] are valid)'.format(optimality))
    
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


def optimize_locally(x0: np.ndarray, symm_mat: np.ndarray, num_strains: int, num_dims: int=6,
                     elastic_tensor_guess=None, optimality: str='D', method: str='COBYLA',
                     tol: float=0.01) -> np.ndarray:
    """
    Locally optimize the strain directions by minimizing a design of experiments
        cost function (i.e. D, A or E optimality)
    :param x0: strains
    :param symm_mat: symmetry matrices for Laue group
    :param num_strains: number of strains gamma
    :param num_dims: number of strain components in each strain
    :param elastic_tensor_guess: guess for elastic constants matrix
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

    args = (symm_mat, num_strains, num_dims, optimality, elastic_tensor_guess)
    D_minimize = spopt.minimize(doe_cost, x0, args=args, method=method,
                                constraints=constraints, options={'maxiter': 200, 'rhobeg': 0.001})
    D_strains = D_minimize.x.reshape(num_strains, num_dims)
    for n, strain in enumerate(D_strains):
        D_strains[n] /= np.linalg.norm(strain)

    return D_strains.reshape(num_strains * num_dims)


def optimize_basin_hopping(x0: np.ndarray, symm_mat: np.ndarray, num_strains: int, num_dims: int=6,
                           elastic_tensor_guess=None, optimality: str='D', method: str='COBYLA',
                           tol: float=0.01, num_itermax: int=1000000, stepsize: float=0.2,
                           temp: float=20.0, orthogonal_step: bool=False,
                           verbose: bool=False) -> np.ndarray:
    def take_normalized_step(x, dims=num_dims, stepsize=stepsize):
        """
        We might be concerned with taking steps that are not isotropic if we
            rescale.
        This however seems not so problematic I would say as anyways the strain
            has his own kind of geometry and it is unclear how to interprete
            homogeneous sampling e.g. in strain space
        """
        x0 = x.copy()

        newx = x0 + (np.random.rand(len(x0)) - 0.5) * stepsize

        D_strains = newx.reshape(int(len(x) / dims), dims)
        for n, strain in enumerate(D_strains):
            D_strains[n] /= np.linalg.norm(strain)

        return D_strains.reshape(len(x))

    def take_normalized_step_ortho(x: np.ndarray, dims: int=num_dims,
                                   stepsize: float=stepsize) -> np.ndarray:
        """
        We might be concerned with taking steps that are not isotropic if we
            rescale.
        This however seems not so problematic I would say as anyways the strain
            has his own kind of geometry and it is
            unclear how to interprete homogeneous sampling e.g. in strain space
        """
        x0 = x.copy()

        newx = x0 + (np.random.rand(len(x0)) - 0.5) * stepsize

        D_strains = newx.reshape(int(len(x) / dims), dims)
        for n, strain in enumerate(D_strains):
            D_strains[n] /= np.linalg.norm(strain)

        Xortho = gram_schmidt_rows(D_strains)
        return Xortho.reshape(len(x))

    # initialize random strain vectors
    for n, strain in enumerate(x0):
        x0[n] /= np.linalg.norm(strain)

    x0 = x0.reshape(num_strains * num_dims)

    constraints = []
    for strain_n in range(num_strains):
        constraints += [{'type': 'ineq',
                         'fun': norm_constraint_lower,
                         'args': (strain_n, num_strains, num_dims, tol)},
                        {'type': 'ineq',
                         'fun': norm_constraint_upper,
                         'args': (strain_n, num_strains, num_dims, tol)}]

    minimizer_kwargs = {'args': (symm_mat, num_strains, num_dims, optimality,
                                 elastic_tensor_guess),
                        'method': method,
                        'constraints': constraints,
                        'options': {'maxiter': 200},
                        'tol': 0.0001}
    if orthogonal_step:
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


def calc_cost(symm_mat: np.ndarray, num_strains: int,
              num_dims: int=6, optimality: str='D'):
    """
    Generate a lambda cost function for a given symmetry, number of strains,
        number of dimensions, and type of optimality (D, A, E)
    :param symm_mat: list of elastic symmetry matrices
    :param num_strains: number of strain directions to optimize over
    :param num_dims: number of dimensions being considered (6 for the 3-D crystal case)
    :param optimality: type of design of experiments optimality
        D for determinate optimization
        A for ...
        E for eigenvalue optimization
    """
    return lambda x0: doe_cost(x0, symm_mat, num_strains, num_dims, optimality)


def basin_hopping(symm_mat: np.ndarray, num_strains: int, num_dims: int=6,
                  elastic_tensor_guess=None, optimality: str='D', method: str='SLSQP',
                  tol: float=0.01, num_itermax: int=1000, step_size: float=0.2,
                  temp: float=10.0, orthogonal: bool=True):
    """
    Generate a lambda basin hopping function for a given system
    :param symm_matrix: list of elastic symmetry matrices
    :param num_strains: number of strain directions to optimize over
    :param num_dims: number of dimensions being considered (6 for the 3-D crystal case)
    :param elastic_tensor_guess: approximate elastic tensor matrix for doing relative value
        optimization
    :param optimality: type of design of experiments optimality
        D for determinate optimization
        A for ...
        E for eigenvalue optimization
    :param method: scipy minimization algorithm
    :param tol: scipy minimization tolerance
    :param num_itermax: scipy maximum number of iterations
    :param step_size: basin hopping step size
    :param temp: basin hopping temperature
    :param orthogonal: whether to take orthogonal steps during basin hopping
    """
    return lambda x0: optimize_basin_hopping(x0, symm_mat, num_strains, num_dims,
                                             elastic_tensor_guess, optimality, method,
                                             tol, num_itermax, step_size, temp, orthogonal)


def generate_olics(laue: str, num_dims: int=6, elastic_tensor_guess=None,
                   optimality: str='D', method: str='SLSQP', 
                   num_itermax: int=1_000, random_tests: int=1_000,
                   basin_tests: int=3, step_size: float=0.1, tol: float=0.001,
                   orthogonalize_strains: bool=True, additional_strains: int=0,
                   auto_temp: bool=True, temp: float=10.0, max_workers: int=4,
                   orthogonal_steps: bool=True, verbose=True, multiprocess: bool=True) -> dict:
    """
    Generate optimal linear-independent coupling strains for a given crystal symmetry, optionally
        using approximate information about the elastic constants of a specific material
    :param laue: Laue symmetry group
    :num_dims: number of dimensions of the voigt-notation elastic tensor
    :elastic_tensor_guess: and approximate elastic tensor for relative error optimization
    :optimality: type of design of experiments optimality ('D', 'E', and 'A' are supported)
    :method: local minimization method for scipy.optimize.minimize,
        method _must_ support constraints
    :num_itermax: maximum number of iterations for local minimization (see scipy.optimize.minimize)
    :param random_tests: number of random sets of strains to generate
    :param basin_tests: number of basin optimization tests to run
    :param step_size: basin hopping step size
    :param tol: tolerance for minimization (see scipy.optimize.minimize)
    :param orthogonalize_strains: whether to orthonormalize each set of strains via QR factorization
        (see numpy.linalg.qr)
    :param additional_strains: number of additional strains beyond the minimum number required by
        symmetry. Additional strains increase the cost of an elastic constants calculation linearly
        but reduce errors at a much higher rate.
    :param auto_temp: automatically determine the basin-hopping temperature through basic
        preliminary error analysis
    :param temp: basin-hopping temperature
    :param max_workers: maximum number of multiprocessing workers used during the optimization
        calculations
    :param orthogonal_steps: take orthogonal steps during basin-hopping optimization
    :param verbose: print progress messages (useful for long-running expensive optimizations)
    """
    symm_mat = get_elastic_symmetries(laue)
    num_strains = get_min_num_strains(laue) + additional_strains
    cost_function = calc_cost(symm_mat, num_strains, num_dims, optimality)
    
    random_strains = []
    
    if verbose:
        print('Running {} ({} strains)'.format(laue, num_strains))
    
    if verbose:
        print('Generating strains ...', end=' ')
        
    for i in range(random_tests * num_strains):
        # generate random strains
        x0 = np.random.rand(num_strains, num_dims)
        
        # normalize each strain
        for n, strain in enumerate(x0):
            x0[n] /= np.linalg.norm(strain)
        
        # orthogonalize the strains
        if orthogonalize_strains:
            x0 = gram_schmidt_rows(x0)
        
        # reduce the dimension of the strain array
        # each strain is now a row instead of a vector
        x0 = x0.reshape(num_strains * num_dims)
        
        random_strains.append(x0)
        
    # reshape the generated strains
    basin_strains = random_strains[:basin_tests*num_strains]
    
    if multiprocess:
        if verbose:
            print('Done')
            print('Initializing multiprocessing ...', end=' ')
            
        # initialize a multiprocessing pool
        try:
            tp = ProcessingPool(max_workers=max_workers)
        except:
            tp.restart()
            
        map_function = tp.map
    else:
        map_function = map
        
    if verbose:
        print('Done')
        print('Calculating costs ...', end=' ')
        
    # calculate costs for random and random orthonormalized strain sets    
    try:
        random_costs = map_function(cost_function, random_strains)
    except Exception as e:
        print(e)
        tp.restart()
        random_costs = map_function(cost_function, random_strains)
    
    if verbose:
        print('Done')
        print('Running basin hopping optimization ...', end=' ')
    
    # set basin hopping temperature
    if auto_temp == True:
        temp = np.percentile(random_costs - min(random_costs), 5)
    else:
        temp = 10.0
        
    # do basin-hopping optimization for random and orthonormalized strains    
    basin_result = basin_hopping(symm_mat=symm_mat,
                                 num_strains=num_strains, num_dims=num_dims,
                                 elastic_tensor_guess=elastic_tensor_guess,
                                 optimality=optimality, method=method, tol=tol,
                                 num_itermax=num_itermax, step_size=step_size,
                                 temp=temp, orthogonal=orthogonal_steps)

    # retrieve results from the basin hopping optimizations
    try:
        optimized_strains = map_function(basin_result, basin_strains)
    except Exception as e:
        print(e)
        tp.restart()
        optimized_strains = map_function(basin_result, basin_strains)
         
    if verbose:
        print('Done')
        print('Calculating costs for basin hopping results ...', end=' ')
            
    # calculate costs for the basin hopping results
    try:
        optimized_costs  = map_function(cost_function, basin_strains)
    except Exception as e:
        print(e)
        tp.restart()
        optimized_costs = map_function(basin_result, basin_strains)
        
    if verbose:
        print('Done')
        print('Preparing outputs ...', end=' ')
        
    # reshape the result strains
    optimized_strains = [ihh.reshape(num_strains, num_dims).tolist() for
                         ihh in optimized_strains]
    
    # return the results for further processing (the top 4 results are saved)
    opt_results = {
        'OLICS': optimized_strains,
        'errors': optimized_costs
    }
    
    result_strains = opt_results['OLICS'][np.argmin(opt_results['errors'])]
    result_cost = min(opt_results['errors'])
    
    if verbose:
        print('Done')
    
    return {'OLICS ({})'.format(laue): result_strains,
            'COST ({})'.format(laue): result_cost}