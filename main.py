import numpy as np
from input import parse_input
from model import IsingModel, model_from_file
from visualization import Visualization
from visualization import visualize_from_file, show_endstate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from time import time


params = parse_input()

N_X = int(params["N_X"])
N_Y = int(params["N_Y"])
MU = params["MU"]
J = params["J"]
BETA = params["BETA"]

BETA = np.log(1 + np.sqrt(2)) / 2

h = np.zeros((N_Y, N_X))
h[0, :]      =  np.ones(h.shape[1])
h[-1, :]     =  np.ones(h.shape[1])
h[0, 40:60]  = -1
h[-1, 40:60] = -1

for sign, ips in zip(["-", "+"], [[0., 1.], [1., 0.]]):
    for NEW_BETA in np.linspace(BETA, .8, 10):

        t_start = time()

        filename = f"beta={NEW_BETA:.2f}_hom{sign}_after100_000.txt"
        model = IsingModel(N_X, N_Y, J=J, BETA=NEW_BETA, MU=MU, h=h, init_p=ips)
        model.run_long_simulation(100_000, 20_000)  # discard simulation up to here

        model.init_spins = model.spins.copy()
        model.change_list = np.empty(0, dtype="int32, int32")

        t_stop = time()
        print(f"Run for BETA={NEW_BETA:.2f} took {t_stop - t_start:.2f}s")

        model.run(10_000, filename=filename)

        endstate = model.calculate_average_endstate(0)

        avg_mag = np.sum(endstate) / (endstate.shape[0] * endstate.shape[1])

        fig, ax = plt.subplots()

        fig.suptitle(f"Average Magnetization: {avg_mag:.3f}")

        im = ax.imshow(endstate)
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        fig.colorbar(im, cax=cax)


        # plt.show()
        fig.savefig(f"beta={NEW_BETA:.2f}_avg_over_10_000_hom{sign}.png", dpi=600)

        t_stop = time()
        print(f"Complete simulation for BETA={NEW_BETA:.2f} took {t_stop - t_start:.2f}s")

