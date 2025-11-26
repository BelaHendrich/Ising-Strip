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

# h = np.zeros((N_Y, N_X))
# h[0, :]      =  np.ones(h.shape[1])
# h[-1, :]     =  np.ones(h.shape[1])
# h[0, 20:50]  = -1
# h[-1, 30:60] = -1

BETA = np.log(1 + np.sqrt(2)) / 2

for dB in np.linspace(-0.3, 0.3, 10):

    # t_start = time()
    #
    # model = IsingModel(N_X, N_Y, J=J, BETA=BETA+dB, MU=MU) # , h=h)
    # model.run_long_simulation(100_000, 10_000, f"no_fields_beta={BETA+dB:.2f}.txt")
    #
    # t_stop = time()
    # print(f"Run for BETA={BETA+dB:.2f} took {t_stop - t_start:.2f}s")


    model = model_from_file(f"no_fields_beta={BETA+dB:.2f}.txt", stop_after=0)
    endstate = model.calculate_average_endstate(90_000)

    avg_mag = np.sum(endstate) / (endstate.shape[0] * endstate.shape[1])

    fig, ax = plt.subplots()

    fig.suptitle(f"Average Magnetization: {avg_mag:.3f}")

    im = ax.imshow(endstate)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    fig.colorbar(im, cax=cax)


    plt.show()
    fig.savefig(f"no_fields_beta={BETA+dB:.2f}.png", dpi=600)

# for i in range(10):
#     model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU, h=h)
#

