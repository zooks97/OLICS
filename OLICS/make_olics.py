from olics import *

tol = 1e-8
num_itermax = 10_000
basin_tests = 4
step_size=1e-2
n_iter = 10
random_tests = 1_000_000
additional_strains=0
max_workers=8
results = {}

sorted_laue = sorted(LAUE_NAMES.keys())
for laue in sorted_laue:
    temp_results = []
    temp_costs = []

    for i in range(n_iter):
        print('Iteration {}'.format(i))
        temp_result = generate_olics(laue, tol=tol, num_itermax=num_itermax, random_tests=random_tests, basin_tests=4, step_size=step_size, additional_strains=additional_strains, max_workers=max_workers)
        temp_results.append(temp_result['OLICS ({})'.format(laue)])
        temp_costs.append(temp_result['COST ({})'.format(laue)])

    result = {
        'OLICS ({})'.format(laue): temp_results[np.argmin(temp_costs)],
        'COST: ({})'.format(laue): temp_costs[np.argmin(temp_costs)]
    }
    results[laue] = result
    display(result)
    
with open('n_plus_one_olics.json', 'w') as f:
    json.dump(results, f)