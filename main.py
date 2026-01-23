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

# copied from script tttmp.py
wetting_temp = 1.4096560829342712
interesting_temps = np.linspace(wetting_temp - 0.1, wetting_temp + 0.1, 10)
int_betas = 1/interesting_temps
int_betas -= 0.11
int_betas = np.linspace(0.50, 0.60, 10)

for sign, ips in zip(["-", "+"], [[1., 0.], [0., 1.]]):
    if sign=="-":
        dists = [i for i in range(24, 30)]
    if sign=="+":
        dists = [i for i in range(15, 23)]
    for N_Y in dists:

        h = np.zeros((N_Y, N_X))
        h[0, :]      =  np.ones(h.shape[1])
        h[-1, :]     =  np.ones(h.shape[1])
        h[0, 40:60]  = -1
        h[-1, 40:60] = -1

        for NEW_BETA in int_betas:
            # --- pick up where I started ---
            # if sign=="-":
            #     continue
            # if sign=="+" and N_Y < 19:
            #     continue
            # if sign=="+" and N_Y == 19 and NEW_BETA <= 0.53:
            #     continue
            print(sign, N_Y, NEW_BETA)
            # --- pick up where I started ---
            t_start = time()

            model = IsingModel(N_X, N_Y, J=J, BETA=NEW_BETA, MU=MU, h=h, init_p=ips)
            _, energies0 = model.run_with_energy(5_000, energy_steps=1)  # discard simulation up to here

            print(model.center_spins())
            print(model.center_line())
            print(model.center_column())

            with open(f"data_{sign}.txt", "a") as f:
                f.write(f"5k/5k {NEW_BETA:.3f} {N_Y}\n")

                f.write(f"{model.center_spins():.3f}\n")
                f.write(f"{model.center_line():.3f}\n")
                f.write(f"{model.center_column():.3f}\n")

            model.init_spins = model.spins.copy()
            model.change_list = np.empty(0, dtype="int32, int32")

            t_stop = time()
            print(f"Run for BETA={NEW_BETA:.2f} took {t_stop - t_start:.2f}s")

            endstate, energies1 = model.run_with_energy(5_000, energy_steps=1)

            energies = np.concatenate((energies0, energies1))

            # ----- Energy Plot -----
            fig, ax = plt.subplots()

            ax.plot(energies, ".")
            ax.axvline(energies0.size)

            ax.set_xlabel("Steps")
            ax.set_ylabel("Energy")
            ax.grid()
            ax.legend()

            fig.savefig(f"beta={NEW_BETA:.2f}_{sign}_{N_Y}_energy.png", dpi=600)
            plt.close(fig)
            # ----- Energy Plot -----

            avg_mag = np.sum(endstate) / (endstate.shape[0] * endstate.shape[1])

            # ----- State Plot -----
            fig, ax = plt.subplots()

            fig.suptitle(f"Average Magnetization: {avg_mag:.3f}")

            im = ax.imshow(endstate)
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="7%", pad="2%")
            fig.colorbar(im, cax=cax)

            fig.savefig(f"beta={NEW_BETA:.2f}_{sign}_{N_Y}.png", dpi=600)
            plt.close(fig)
            # ----- State Plot -----

            t_stop = time()
            print(f"Complete simulation for BETA={NEW_BETA:.2f} took {t_stop - t_start:.2f}s")

# NOTE: Remember this
exit()

# ===== negative state =====
# ===== negative state =====
# ===== negative state =====

# change below

# copied from script tttmp.py
wetting_temp = 1.4096560829342712
interesting_temps = np.linspace(wetting_temp - 0.1, wetting_temp + 0.1, 10)
int_betas = 1/interesting_temps
int_betas -= 0.11
int_betas = np.linspace(0.40, 0.75, 7)

# for sign, ips in zip(["-", "+"], [[1., 0.], [0., 1.]]):
for N_Y in [19, 18, 17]:

    h = np.zeros((N_Y, N_X))
    h[0, :]      =  np.ones(h.shape[1])
    h[-1, :]     =  np.ones(h.shape[1])
    h[0, 40:60]  = -1
    h[-1, 40:60] = -1

    sign, ips = "-", [1., 0.]


    for NEW_BETA in int_betas:
        t_start = time()

        filename = f"LATESTSWEEP_beta={NEW_BETA:.2f}_hom{sign}_after30_000.txt"
        model = IsingModel(N_X, N_Y, J=J, BETA=NEW_BETA, MU=MU, h=h, init_p=ips)
        _, energies0 = model.run_with_energy(5_000)  # discard simulation up to here

        print(model.center_spins())
        print(model.center_line())
        print(model.center_column())

        model.init_spins = model.spins.copy()
        model.change_list = np.empty(0, dtype="int32, int32")

        t_stop = time()
        print(f"Run for BETA={NEW_BETA:.2f} took {t_stop - t_start:.2f}s")

        endstate, energies1 = model.run_with_energy(5_000)

        energies = np.concatenate((energies0, energies1))

        # ----- Energy Plot -----
        fig, ax = plt.subplots()

        ax.plot(energies, ".")
        ax.axvline(energies0.size)

        ax.set_xlabel("Steps [10]")
        ax.set_ylabel("Energy")
        ax.grid()
        ax.legend()

        fig.savefig(f"LATESTSWEEP_NY={N_Y}_smaller_h_beta={NEW_BETA:.2f}_energy.png", dpi=600)
        plt.close(fig)
        # ----- Energy Plot -----

        avg_mag = np.sum(endstate) / (endstate.shape[-1] * endstate.shape[1])

        # ----- State Plot -----
        fig, ax = plt.subplots()

        fig.suptitle(f"Average Magnetization: {avg_mag:.3f}")

        im = ax.imshow(endstate)
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        fig.colorbar(im, cax=cax)

        fig.savefig(f"LATESTSWEEP_NY={N_Y}_smaller_h_beta={NEW_BETA:.2f}_avg_over_30_000_hom{sign}.png", dpi=600)
        plt.close(fig)
        # ----- State Plot -----

        t_stop = time()
        print(f"Complete simulation for BETA={NEW_BETA:.2f} took {t_stop - t_start:.2f}s")

