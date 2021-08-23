# %%
import numpy as np
import elastic

# %%
n_alpha = 3
M_bar = elastic.get_elastic_symmetries('CI')
c = np.array([104, 73, 32])
# %%
n_gamma = 2
eta = elastic.get_ulics('HI')
# %%
E_bar = np.vstack(np.matmul(M_bar, eta.reshape(n_gamma, 6).T).T)
# %%
