import numpy as np
from input import parse_input
from model import IsingModel
from visualization import Visualization
from visualization import visualize_from_file, show_endstate
import matplotlib.pyplot as plt


params = parse_input()

N_X = int(params["N_X"])
N_Y = int(params["N_Y"])
MU = params["MU"]
J = params["J"]
BETA = params["BETA"]
print(BETA)

# model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU)
# vis   = Visualization(model, with_energy=False, with_mag=False, stepsize=100)

# ani = vis.animate(interval=16)
# plt.show()

for i in range(10):
    h = np.zeros((N_Y, N_X))
    h[0, :]      =  np.ones(h.shape[1])
    h[-1, :]     =  np.ones(h.shape[1])
    h[0, 20:40]  = -1
    h[-1, 30:50] = -1

    model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU, h=h)

    model.run(25_000, f"test_{i}.txt")
    N_Y += 5
    # show_endstate(f"test_{i}.txt")
# visualize_from_file("test.txt", N_X, N_Y)
