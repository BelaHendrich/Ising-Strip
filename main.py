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

N_Y = 13
for _ in range(10):

    h = np.zeros((N_Y, N_X))
    h[0, :]      =  np.ones(h.shape[1])
    h[-1, :]     =  np.ones(h.shape[1])
    h[0, 30:60]  = -1
    h[-1, 30:60] = -1

    t_start = time()

    filename = f"ny={N_Y}_hom.txt"
    model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU, h=h, init_p=[0., 1.])
    model.run_long_simulation(110_000, 20_000, filename)

    t_stop = time()
    print(f"Run for N_Y={N_Y} took {t_stop - t_start:.2f}s")

    model = model_from_file(filename,
                            stop_after=100_000,
                            keep_change_list=False)

    skips = 100_000 * model.N_X * model.N_Y + 10
    model.change_list = np.loadtxt(f"Data/{filename}",
                                   dtype="int32",
                                   skiprows=skips,
                                   delimiter=',')

    filename = f"ny={N_Y}_hom_after100_000.txt"
    model.write_params_to_file(filename)
    model.write_state_to_file(filename)

    endstate = model.calculate_average_endstate(0)

    avg_mag = np.sum(endstate) / (endstate.shape[0] * endstate.shape[1])

    fig, ax = plt.subplots()

    fig.suptitle(f"Average Magnetization: {avg_mag:.3f}")

    im = ax.imshow(endstate)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    fig.colorbar(im, cax=cax)


    # plt.show()
    fig.savefig(f"fields_ny={N_Y}_avg_over_10_000_hom.png", dpi=600)

    t_stop = time()
    print(f"Complete simulation for N_Y={N_Y} took {t_stop - t_start:.2f}s")

    N_Y += 1

N_Y = 13
for _ in range(10):

    h = np.zeros((N_Y, N_X))
    h[0, :]      =  np.ones(h.shape[1])
    h[-1, :]     =  np.ones(h.shape[1])
    h[0, 30:60]  = -1
    h[-1, 30:60] = -1

    t_start = time()

    model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU, h=h)
    model.run_long_simulation(110_000, 20_000, f"ny={N_Y}.txt")

    t_stop = time()
    print(f"Run for N_Y={N_Y} took {t_stop - t_start:.2f}s")

    filename = f"ny={N_Y}.txt"
    model = model_from_file(filename,
                            stop_after=100_000,
                            keep_change_list=False)

    skips = 100_000 * model.N_X * model.N_Y + 10
    model.change_list = np.loadtxt(f"Data/{filename}",
                                   dtype="int32",
                                   skiprows=skips,
                                   delimiter=',')

    filename = f"ny={N_Y}_after100_000.txt"
    model.write_params_to_file(filename)
    model.write_state_to_file(filename)

    endstate = model.calculate_average_endstate(0)

    avg_mag = np.sum(endstate) / (endstate.shape[0] * endstate.shape[1])

    fig, ax = plt.subplots()

    fig.suptitle(f"Average Magnetization: {avg_mag:.3f}")

    im = ax.imshow(endstate)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    fig.colorbar(im, cax=cax)


    # plt.show()
    fig.savefig(f"fields_ny={N_Y}_avg_over_10_000.png", dpi=600)

    t_stop = time()
    print(f"Complete simulation for N_Y={N_Y} took {t_stop - t_start:.2f}s")

    N_Y += 1
