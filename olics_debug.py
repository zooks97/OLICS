from OLICS import generate_olics

tol = 1e-1
num_itermax = 10_000
basin_tests = 2
step_size=1e-1
n_iter = 1
random_tests = 100
additional_strains = 1
max_workers = 1
multiprocess = True
laue = 'N'

result = generate_olics(laue, tol=tol, num_itermax=num_itermax,
                        random_tests=random_tests, basin_tests=basin_tests,
                        step_size=step_size, additional_strains=additional_strains,
                        max_workers=max_workers, multiprocess=multiprocess)