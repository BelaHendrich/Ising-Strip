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
DT      = "int32"       # data type for indices
DTYPE   = f"{DT}, {DT}"


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

    @classmethod
    def from_spins(Self, spins, *args, **kwargs):

        model = Self(*args, **kwargs)

        model.spins = spins
        model.init_spins = model.spins.copy()

        return model

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

        return (-1, -1)

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

    def run(self, steps, filename=None, append_file=False):
        iterations = self.N_X * self.N_Y * steps

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        task = progress.add_task("Simulation", total=iterations)

        if filename and not append_file:
            self.check_for_file(filename)
            self.write_params_to_file(filename)

        i_list = np.random.randint(self.N_Y, size=iterations)
        j_list = np.random.randint(self.N_X, size=iterations)
        probs  = np.random.uniform(size=iterations)

        new_changes = np.empty(iterations, dtype=DTYPE)

        with progress:
            for idx, (i, j, p) in enumerate(zip(i_list, j_list, probs)):
                res = self.update(i, j, p)
                new_changes[idx] = res
                progress.update(task, advance=1)

        self.change_list = np.concatenate((self.change_list, new_changes))

        if filename:
            self.write_change_list_to_file(new_changes, filename)

        return

    def run_long_simulation(self, steps, chunks, filename):
        '''
        Break up the simulation into a succession of smaller ones.
        This avoids running out of memory when trying to precompute
        all indices and probabilities.
        '''
        steps_taken = 0
        append_file = False
        while steps_taken < steps - chunks:
            self.run(chunks, filename=filename, append_file=append_file)
            # reset saved change_list to save memory
            self.change_list = np.empty(0, dtype=DTYPE)
            steps_taken += chunks
            append_file = True

        self.run(steps - steps_taken, filename=filename,
                 append_file=append_file)

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
        '''
        cutoff = self.N_X * self.N_Y * cutoff  # iterations = area * steps

        current_state = self.init_spins.copy()
        avg_endstate = np.zeros_like(current_state)

        for (i, j) in self.change_list[:cutoff]:
            if i >= 0:
                current_state[i, j] *= -1

        for (i, j) in self.change_list[cutoff:]:
            if i >= 0:
                current_state[i, j] *= -1
            avg_endstate += current_state

        averaging_length = len(self.change_list) - cutoff

        return avg_endstate / averaging_length


def model_from_file(filename, stop_after=None, keep_change_list=False):
    '''
    Creates a new model with the same parameters as in @filename.
    The initial spin configuration is the endstate of the one specified
    in @filename.
    '''
    filename = OUT_DIR + filename

    params = np.loadtxt(filename, dtype="str", max_rows=10, delimiter='=')
    params = dict(np.strings.strip(params))

    N_X = int(params["N_X"])
    N_Y = int(params["N_Y"])

    J    = float(params["J"])
    BETA = float(params["BETA"])
    MU   = float(params["MU"])

    fields = params["h"]
    h = [1 if d == "+" else -1 if d == "-" else 0 for d in fields]
    h = np.array(h)
    h.resize(N_Y, N_X)

    boundary_x = params["boundary_x"]
    boundary_y = params["boundary_y"]

    spins = params["spins"]
    current_state = [1 if d == "+" else -1 for d in spins]
    current_state = np.array(current_state)
    current_state.resize(N_Y, N_X)

    if stop_after is None:
        change_list = np.loadtxt(filename, dtype=DT, skiprows=10,
                                 delimiter=',')
    else:
        stop_after = N_X * N_Y * stop_after  # iterations = area * steps
        change_list = np.loadtxt(filename, dtype=DT, skiprows=10,
                                 max_rows=stop_after, delimiter=',')

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )
    task = progress.add_task("Reevaluation", total=change_list.shape[0])

    with progress:
        for i, j in change_list:
            if i >= 0:
                current_state[i, j] *= -1
            progress.update(task, advance=1)

    model = IsingModel.from_spins(current_state, N_X, N_Y,
                                  BOUNDARIES=(boundary_x, boundary_y),
                                  J=J, BETA=BETA, MU=MU, h=h)
    if keep_change_list:
        model.change_list = change_list

    return model

