import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from time import time
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)


OUT_DIR = "Data/"


class IsingModel():
    def __init__(self, N_X, N_Y, BOUNDARIES=("cyclic", "open"),
                 J=1, BETA=1, MU=1, h=None):
        self.N_X = N_X
        self.N_Y = N_Y

        self.J = J
        self.BETA = BETA
        self.MU = MU

        self.h = h
        if h is None:
            self.h = np.zeros((N_Y, N_X))

        self.boundary_x = BOUNDARIES[0]
        self.boundary_y = BOUNDARIES[1]

        self.spins = np.random.choice([-1, 1], (N_Y, N_X),
                                      p=[.25, .75])

    def hamiltonian(self):
        # if not h:
        #     h = np.zeros_like(self.spins)
        #     h[0, :]      =  np.ones(self.N_X)
        #     h[-1, :]     =  np.ones(self.N_X)
        #     h[0, 20:40]  = -1
        #     h[-1, 30:50] = -1

        spins_right = np.roll(self.spins, +1, axis=1)
        spins_left  = np.roll(self.spins, -1, axis=1)

        if self.boundary_x[0] == 'o':  # open boundaries
            spins_right[:, 0] = np.zeros(self.N_Y)
            spins_left[:, -1] = np.zeros(self.N_Y)

        spins_down  = np.roll(self.spins, +1, axis=0)
        spins_up    = np.roll(self.spins, -1, axis=0)

        if self.boundary_y[0] == 'o':  # open boundaries
            spins_down[0, :] = np.zeros(self.N_X)
            spins_up[-1, :]  = np.zeros(self.N_X)

        shifted_spins = spins_right + spins_left + spins_down + spins_up

        return -np.sum(self.J*self.spins*shifted_spins) - \
               self.MU*np.sum(self.h*self.spins)

    def energy_diff(self, i, j):
        spin_state = self.spins
        delta_e = spin_state[i, (j+1) % self.N_X] + \
                  spin_state[i, (j-1) % self.N_X] + \
                  spin_state[(i+1) % self.N_Y, j] + \
                  spin_state[(i-1) % self.N_Y, j]

        if self.boundary_x[0] == 'o':  # open
            if j == 0:         # remove spin to the left
                delta_e -= spin_state[i, (j-1) % self.N_X]
            if j == self.N_Y:  # remove spin to the right
                delta_e -= spin_state[i, (j+1) % self.N_X]

        if self.boundary_y[0] == 'o':  # open
            if i == 0:         # remove spin above
                delta_e += spin_state[(i-1) % self.N_Y, j]
            if i == self.N_X:  # remove spin below
                delta_e += spin_state[(i+1) % self.N_Y, j]

        delta_e *= self.J
        delta_e += self.MU * self.h[i, j]

        return 2 * delta_e * spin_state[i, j]

    def magnetization(self):
        return np.sum(self.spins)

    def update(self, i, j, p):
        delta_e = self.energy_diff(i, j)

        if delta_e < 0:
            self.spins[i, j] *= -1
            return (i, j)

        transition_prob = np.exp(-self.BETA*delta_e)
        if transition_prob > p:
            self.spins[i, j] *= -1
            return (i, j)

        return

    def large_steps(self, step_size=1_000):
        for _ in range(step_size):
            self.update()
        return

    def write_state_to_file(self, filename):
        filename = OUT_DIR + filename
        with open(filename, "a") as f:
            data = "".join(map(lambda x: "+" if x > 0 else "-",
                               list(self.spins.flatten())))
            f.write(data + "\n")

        return

    def write_change_to_file(self, filename, i, j):
        filename = OUT_DIR + filename
        with open(filename, "a") as f:
            data = f"{i},{j}"
            f.write(data + "\n")

        return

    def write_params_to_file(self, filename):
        filename = OUT_DIR + filename
        with open(filename, "w") as f:
            data = vars(self)
            for k, v in data.items():
                if k == "h" or k == "spins":
                    v = "".join(map(lambda x: "+" if x > 0 else
                                    "-" if x < 0 else "0",
                                    list(v.flatten())))
                f.write(f"{k} = {v}\n")

        return

    def check_for_file(self, filename):
        if filename in os.listdir(OUT_DIR):
            user_in = input(f"File {filename} already exists in ./{OUT_DIR}." +
                             " Do you want to overwrite it? [Y/n]")
            if user_in and user_in.lower()[0] == "n":
                print("Aborting.")
                sys.exit()

        return

    def run(self, steps, filename):
        iterations = self.N_X * self.N_Y * steps

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        task = progress.add_task("Simulation", total=iterations)

        self.check_for_file(filename)
        self.write_params_to_file(filename)

        i_list = np.random.randint(self.N_Y, size=iterations)
        j_list = np.random.randint(self.N_X, size=iterations)
        probs  = np.random.uniform(size=iterations)

        with progress:
            for i, j, p in zip(i_list, j_list, probs):
                res = self.update(i, j, p)
                if res:
                    self.write_change_to_file(filename, res[0], res[1])
                progress.update(task, advance=1)

        return

