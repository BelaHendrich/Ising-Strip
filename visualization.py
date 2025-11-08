import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import IsingModel


class Visualization():
    def __init__(self, model, with_energy=False, with_mag=False, stepsize=None):
        num_plots = 1 + int(with_energy) + int(with_mag)
        ratio_list = [5, 1, 1]
        gridspec_kw = {'height_ratios': ratio_list[:num_plots]}

        self.fig, self.ax = plt.subplots(num_plots, 1, gridspec_kw=gridspec_kw)
        self.model = model

        if num_plots>1:
            self.im = self.ax[0].imshow(model.spins)
        else:
            self.im = self.ax.imshow(model.spins)

        energies = self.model.history["energies"]
        magnetizations = self.model.history["magnetizations"]

        self.e_lines, self.m_lines = None, None
        if with_energy:
            self.e_lines = self.ax[1].plot(energies)[0]
        if with_mag:
            self.m_lines = self.ax[2].plot(magnetizations)[0]

        self.stepsize = stepsize

    def update(self, frame):
        if self.stepsize:
            self.model.large_steps(step_size=self.stepsize)
        else:
            self.model.update()

        energies = self.model.history["energies"]
        magnetizations = self.model.history["magnetizations"]
        iterations = np.arange(len(energies))

        if self.e_lines:
            self.ax[1].set(xlim=[0, iterations[-1]],
                           ylim=[np.min(energies), np.max(energies)])
            self.e_lines.set_xdata(iterations)
            self.e_lines.set_ydata(energies)
        if self.m_lines:
            self.ax[2].set(xlim=[0, iterations[-1]],
                           ylim=[np.min(magnetizations), np.max(magnetizations)])
            self.m_lines.set_xdata(iterations)
            self.m_lines.set_ydata(magnetizations)

        self.im.set_data(self.model.spins)
        self.fig.suptitle(frame)

        return (self.im, self.e_lines, self.m_lines)

    def animate(self, interval=16, num_frames=40):
        ani = animation.FuncAnimation(fig=self.fig,
                                      func=lambda f: self.update(f),
                                      frames=num_frames,
                                      interval=interval)
        return ani


def visualize_from_file(filename, n_x, n_y):
    filename = "Data/" + filename

    current_state = []
    change_list   = []

    with open(filename, "r") as f:
        data = f.readlines()

        current_state = [1 if d == "+" else -1 for d in data[0]]
        # N = int(len(current_state)**0.5)
        current_state = np.array(current_state)
        current_state.resize(n_y, n_x)

        change_list = data[1:]

    fig, ax = plt.subplots()

    im = ax.imshow(current_state)

    def file_update(frame):
        indices = change_list[frame].split(",")
        i, j = int(indices[0]), int(indices[1])

        current_state[i, j] *= -1

        im.set_data(current_state)
        fig.suptitle(frame)

        return im


    ani = animation.FuncAnimation(fig=fig, func=file_update,
                                  frames=len(change_list), interval=.01)
    plt.show()


def show_endstate(filename, n_x, n_y):
    filename = "Data/" + filename

    with open(filename, "r") as f:
        data = f.readlines()

        current_state = [1 if d == "+" else -1 for d in data[0]]
        # N = int(len(current_state)**0.5)
        current_state = np.array(current_state)
        current_state.resize(n_y, n_x)

        change_list = data[1:]

    for change in change_list:
        indices = change.split(",")
        i, j = int(indices[0]), int(indices[1])
        current_state[i, j] *= -1

    fig, ax = plt.subplots(3, 1)
    h = np.zeros((n_y, n_x))
    h[0, :]      =  np.ones(h.shape[1])
    h[-1, :]     =  np.ones(h.shape[1])
    h[0, 20:40]  = -1
    h[-1, 30:50] = -1

    im_0 = ax[0].imshow(current_state)
    im_1 = ax[1].imshow(h)
    im_2 = ax[2].imshow(h*current_state)
    fig.colorbar(im_0, ax=ax[0])
    fig.colorbar(im_1, ax=ax[1])
    fig.colorbar(im_2, ax=ax[2])
    plt.show()

    return

