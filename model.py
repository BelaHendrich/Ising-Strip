import numpy as np
import matplotlib.pyplot as plt
import sys
import os
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
                 J=1, BETA=1, MU=1, h=0):
        self.N_X = N_X
        self.N_Y = N_Y

        self.J = J
        self.BETA = BETA
        self.MU = MU
        self.h = h  # external field, same dimension as spins

        self.boundary_x = BOUNDARIES[0]
        self.boundary_y = BOUNDARIES[1]

        self.spins = np.random.choice([-1, 1], (N_Y, N_X))

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
        delta_e = self.spins[i, (j+1) % self.N_X] + \
                  self.spins[i, (j-1) % self.N_X] + \
                  self.spins[(i+1) % self.N_Y, j] + \
                  self.spins[(i-1) % self.N_Y, j]

        if self.boundary_x[0] == 'o':  # open
            if j == 0:         # remove spin to the left
                delta_e -= self.spins[i, (j-1) % self.N_X]
            if j == self.N_Y:  # remove spin to the right
                delta_e -= self.spins[i, (j+1) % self.N_X]

        if self.boundary_y[0] == 'o':  # open
            if i == 0:         # remove spin above
                delta_e += self.spins[(i-1) % self.N_Y, j]
            if i == self.N_X:  # remove spin below
                delta_e += self.spins[(i+1) % self.N_Y, j]

        delta_e *= self.J
        delta_e += self.MU * self.h[i, j]

        return 2 * delta_e * self.spins[i, j]

    def magnetization(self):
        return np.sum(self.spins)

    def update(self):
        while True:
            i = np.random.randint(self.N_Y)
            j = np.random.randint(self.N_X)

            delta_e = self.energy_diff(i, j)
            print(f"energy_diff: {delta_e}")

            if delta_e < 0:
                transition_prob = 1
                print("yello", transition_prob)
            else:
                transition_prob = np.exp(-self.BETA*(delta_e))
                print("hello", transition_prob)

            if transition_prob > np.random.uniform():
                self.spins[i, j] *= -1
                print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                break

        return (i, j)

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

    def run(self, iterations, filename):
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        task = progress.add_task("Simulation", total=iterations)
        self.check_for_file(filename)
        self.write_params_to_file(filename)
        # self.write_state_to_file(filename)
        with progress:
            for _ in range(iterations):
                i, j = self.update()
                self.write_change_to_file(filename, i, j)
                progress.update(task, advance=1)

        return

