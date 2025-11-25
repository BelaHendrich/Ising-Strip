import numpy as np
from input import parse_input
from model import IsingModel, model_from_file
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

# h = np.zeros((N_Y, N_X))
# h[0, :]      =  np.ones(h.shape[1])
# h[-1, :]     =  np.ones(h.shape[1])
# h[0, 20:50]  = -1
# h[-1, 30:60] = -1

# BETA = np.log(1 + np.sqrt(2)) / 2
# model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU) # , h=h)
# model.run_long_simulation(1_500_000, 50_000, "long_test_no_fields.txt")

model = model_from_file("state_after400M.txt")
model.change_list = np.loadtxt("Data/long_test_no_fields.txt",
                               dtype="uint16",
                               skiprows=400_000_010,
                               delimiter=',')

endstate = model.calculate_average_endstate(0)
avg_mag = np.sum(endstate) / (endstate.shape[0] * endstate.shape[1])

fig, ax = plt.subplots()

fig.suptitle(f"Average Magnetization: {avg_mag:.3f}")

im = ax.imshow(endstate)
fig.colorbar(im, ax=ax)

plt.show()
fig.savefig("average_endstate_without_fields.png", dpi=600)

# for i in range(10):
#     model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU, h=h)
#
#     t_start = time()
#     # model.find_num_of_steps(100)
#     model.run(1000, f"test_{i}.txt")
#     t_stop = time()
    # print(f"Run for BETA={BETA:.2f} took {t_stop - t_start:.2f}s")

