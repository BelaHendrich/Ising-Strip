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
BETA = params["BETA"]

h = np.zeros((N_Y, N_X))
h[0, :]      =  np.ones(h.shape[1])
h[-1, :]     =  np.ones(h.shape[1])
h[0, 20:50]  = -1
h[-1, 30:60] = -1

BETA = 0.45
model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU, h=h)
model.find_num_of_steps(1000, cutoff=500_000)

# for i in range(10):
#     model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU, h=h)
#
#     t_start = time()
#     # model.find_num_of_steps(100)
#     model.run(1000, f"test_{i}.txt")
#     t_stop = time()
    # print(f"Run for BETA={BETA:.2f} took {t_stop - t_start:.2f}s")

