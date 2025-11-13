import numpy as np
import matplotlib.pyplot as plt
import visualization


def get_endstate(filename):
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

        change_list = data[9:]

    for change in change_list:
        indices = change.split(",")
        i, j = int(indices[0]), int(indices[1])
        current_state[i, j] *= -1

    return current_state, h


def parse_params(filename):
    filename = "Data/" + filename

    parameters = {"N_X": None,
                  "N_Y": None,
                  "J": None,
                  "BETA": None,
                  "MU": None,
                  }

    with open(filename, "r") as f:
        for line in f.readlines()[:5]:
            line = line.split()
            if line[0] not in parameters.keys():
                raise ValueError(f"The parameters in {filename} " +
                                 f"must be in {list(parameters.keys())}!")
            parameters[line[0]] = float(line[2])

    for key, value in parameters.items():
        if value is None:
            raise ValueError(f"The value of {key} is missing!")

    return parameters


def show_widths():
    fig, ax = plt.subplots(5, 2, sharex=True)

    for i in range(10):
        filename = f"vary_width_{i}.txt"

        current_state, _ = get_endstate(filename)

        i_x, i_y = i%5, i//5
        last_im = ax[i_x, i_y].imshow(current_state)
        ax[i_x, i_y].set_title(f"width={current_state.shape[0]}")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(last_im, cax=cbar_ax)

    plt.show()
    fig.savefig("different_widths.png", dpi=600)


def show_betas():
    fig, ax = plt.subplots(10, 2, sharex=True, sharey=True)

    for i in range(20):
        filename = f"vary_beta_{i}.txt"

        current_state, _ = get_endstate(filename)
        params = parse_params(filename)
        BETA = params["BETA"]

        i_x, i_y = i%10, i//10
        last_im = ax[i_x, i_y].imshow(current_state)
        ax[i_x, i_y].set_title(f"beta={BETA:.2f}")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(last_im, cax=cbar_ax)

    plt.show()
    fig.savefig("different_betas.png", dpi=600)


if __name__ == "__main__":
    show_betas()
