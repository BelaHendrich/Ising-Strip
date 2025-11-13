import numpy as np
from input import parse_input
from model import IsingModel
from visualization import Visualization
from visualization import visualize_from_file, show_endstate
import matplotlib.pyplot as plt
from time import time


params = parse_input()

N_X = int(params["N_X"])
N_Y = int(params["N_Y"])
MU = params["MU"]
J = params["J"]
# BETA = params["BETA"]

# model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU)
# vis   = Visualization(model, with_energy=False, with_mag=False, stepsize=100)

# ani = vis.animate(interval=16)
# plt.show()

for i, BETA in enumerate(np.linspace(0.1, 0.9, 20)):
    h = np.zeros((N_Y, N_X))
    h[0, :]      =  np.ones(h.shape[1])
    h[-1, :]     =  np.ones(h.shape[1])
    h[0, 20:50]  = -1
    h[-1, 30:60] = -1

    model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU, h=h)

    t_start = time()
    model.run(300_000, f"vary_beta_{i}.txt")
    t_stop = time()
    print(f"Run {i} took {t_stop - t_start:.2f}s")

BETA = 0.8

for i, N_Y in enumerate(range(10, 21)):
    h = np.zeros((N_Y, N_X))
    h[0, :]      =  np.ones(h.shape[1])
    h[-1, :]     =  np.ones(h.shape[1])
    h[0, 20:50]  = -1
    h[-1, 30:60] = -1

    model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU, h=h)

    t_start = time()
    model.run(100_000, f"vary_width_{i}.txt")
    t_stop = time()
    print(f"Run {i} took {t_stop - t_start:.2f}s")
