import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from model import IsingModel


class Visualization():
    def __init__(self, model, with_energy=False, with_mag=False):
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

    def update(self, frame):
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


def visualize_from_file(filename):
    filename = "Data/" + filename

    current_state = []
    change_list   = []

    with open(filename, "r") as f:
        data = f.readlines()

        n_x = int(data[0].split()[2])
        n_y = int(data[1].split()[2])

        spins = data[8].split()[2]  # spins = [data]
        current_state = [1 if d == "+" else -1 for d in spins]
        current_state = np.array(current_state)
        current_state.resize(n_y, n_x)

        # ignore change_list
        change_list = data[10:]

    fig, ax = plt.subplots()

    im = ax.imshow(current_state)

    def file_update(frame):
        indices = change_list[frame].split(",")
        i, j = int(indices[0]), int(indices[1])

        if i > 0:
            current_state[i, j] *= -1

        im.set_data(current_state)
        fig.suptitle(frame)

        return im


    ani = animation.FuncAnimation(fig=fig, func=file_update,
                                  frames=len(change_list), interval=.01)
    plt.show()


def show_endstate(filename):
    filename = "Data/" + filename

    with open(filename, "r") as f:
        data = f.readlines()

        n_x = int(data[0].split()[2])
        n_y = int(data[1].split()[2])

        fields = data[5].split()[2]  # h = [data]
        h = [1 if d == "+" else -1 if d == "-" else 0 for d in fields]
        h = np.array(h)
        h.resize(n_y, n_x)

        # ignore 8 lines of parameters
        spins = data[8].split()[2]  # spins = [data]
        current_state = [1 if d == "+" else -1 for d in spins]
        current_state = np.array(current_state)
        current_state.resize(n_y, n_x)

        # ignore change_list
        change_list = data[10:]

    for change in change_list:
        indices = change.split(",")
        i, j = int(indices[0]), int(indices[1])
        if i > 0:
            current_state[i, j] *= -1

    fig, ax = plt.subplots(3, 1)

    im_0 = ax[0].imshow(current_state)
    im_1 = ax[1].imshow(h)
    im_2 = ax[2].imshow(h*current_state)

    ims = [im_0, im_1, im_2]

    for i, im in enumerate(ims):
        ax_divider = make_axes_locatable(ax[i])
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        fig.colorbar(im, cax=cax)

    plt.show()

    return

