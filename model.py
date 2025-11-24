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
DTYPE   = ("uint16, uint16")  # data type for change lists


class IsingModel():
    def __init__(self, N_X, N_Y, BOUNDARIES=("cyclic", "open"),
                 J=1, BETA=1, MU=1, h=None, init_p=[.5, .5]):
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
                                      p=init_p)

        # Will contain the list of all flipped spins
        self.change_list = np.empty(0, dtype=DTYPE)
        self.init_spins = self.spins.copy()

    def from_spins(self, N_X, N_Y, spins, BOUNDARIES=("cyclic", "open"),
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

        self.spins = spins

        # Will contain the list of all flipped spins
        self.change_list = np.empty(0, dtype=DTYPE)
        self.init_spins = self.spins.copy()

        return self

    def num_updates(self):
        return len(self.change_list)

    def hamiltonian(self):
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

    def write_change_list_to_file(self, change_list, filename):
        filename = OUT_DIR + filename
        with open(filename, "a") as f:
            for (i, j) in change_list:
                f.write(f"{i},{j}" + "\n")

        return

    def write_params_to_file(self, filename):
        filename = OUT_DIR + filename
        with open(filename, "w") as f:
            data = vars(self)
            for k, v in data.items():
                if k == "init_spins":
                    continue  # don't write the initial state twice
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

    def run(self, steps, filename=None):
        iterations = self.N_X * self.N_Y * steps

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        task = progress.add_task("Simulation", total=iterations)

        if filename:
            self.check_for_file(filename)
            self.write_params_to_file(filename)

        i_list = np.random.randint(self.N_Y, size=iterations)
        j_list = np.random.randint(self.N_X, size=iterations)
        probs  = np.random.uniform(size=iterations)

        new_changes = np.empty(iterations, dtype=DTYPE)
        idx = 0

        with progress:
            for i, j, p in zip(i_list, j_list, probs):
                res = self.update(i, j, p)
                if res:
                    new_changes[idx] = res
                    idx += 1
                progress.update(task, advance=1)

        new_changes = new_changes[:idx]
        self.change_list = np.concatenate((self.change_list, new_changes))

        if filename:
            self.write_change_list_to_file(new_changes, filename)

        return

    def find_num_of_steps(self, resolution, error_margin=.05, cutoff=100_000):
        '''
        Start with a random configuration with no surface fields
        and wait till the spin-distribution is homogeneous.
        @resolution is the number of steps after which the
        magnetization is checked.
        @error_margin is the percentage of spins that need not
        be aligned.
        @cutoff is the maximum number of steps this test-model will take.
        '''
        test_model = IsingModel(self.N_X, self.N_Y, J=self.J,
                                BETA=self.BETA, MU=self.MU)

        mags = [test_model.magnetization()]

        steps_taken = 0
        target_magnetization = (self.N_X * self.N_Y) * (1 - error_margin)
        while np.abs(mags[-1]) < target_magnetization and steps_taken < cutoff:
            test_model.run(resolution)
            steps_taken += resolution
            mags.append(test_model.magnetization())
            print(mags[-1], target_magnetization, steps_taken)

        return steps_taken

    def calculate_average_endstate(self, cutoff):
        '''
        Average all states after same @cutoff.
        NOTE: @cutoff is an index in the change_list, it does
              not correspond to the number of steps!
        '''
        current_state = self.init_spins.copy()
        avg_endstate = np.zeros_like(current_state)

        for (i, j) in self.change_list[:cutoff]:
            current_state[i, j] *= -1

        for (i, j) in self.change_list[cutoff:]:
            current_state[i, j] *= -1
            avg_endstate += current_state

        averaging_length = len(self.change_list) - cutoff

        return avg_endstate / averaging_length


def model_from_file(filename):
    '''
    Creates a new model with the same parameters as in @filename.
    The initial spin configuration is the endstate of the one specified
    in @filename.
    '''
    filename = OUT_DIR + filename

    with open(filename, "r") as f:
        data = f.readlines()

        n_x = int(data[0].split()[2])
        n_y = int(data[1].split()[2])

        j    = float(data[2].split()[2])
        beta = float(data[3].split()[2])
        mu   = float(data[4].split()[2])

        fields = data[5].split()[2]  # h = [data]
        h = [1 if d == "+" else -1 if d == "-" else 0 for d in fields]
        h = np.array(h)
        h.resize(n_y, n_x)

        boundary_x = data[6].split()[2].strip()
        boundary_y = data[7].split()[2].strip()

        spins = data[8].split()[2]  # spins = [data]
        current_state = [1 if d == "+" else -1 for d in spins]
        current_state = np.array(current_state)
        current_state.resize(n_y, n_x)

        # ignore change_list
        change_list = data[10:]

    for change in change_list:
        indices = change.split(",")
        i, j = int(indices[0]), int(indices[1])
        current_state[i, j] *= -1

    model = IsingModel.from_spins(n_x, n_y, spins,
                                  BOUNDARIES=(boundary_x, boundary_y),
                                  J=j, BETA=beta, MU=mu, h=h)

    return model

