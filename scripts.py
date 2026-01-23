import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import visualization
from input import parse_input
from model import IsingModel


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
        if i >= 0:
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


def find_num_of_steps(model, resolution, error_margin=.05):
    '''Start with a random configuration with no surface fields
       and wait till the spin-distribution is homogeneous.
       @resolution is the number of steps after which the
       magnetization is checked
       @error_margin is the percentage of spins that need not
       be aligned.'''
    test_model = IsingModel(model.N_X, model.N_Y, J=model.J,
                            BETA=model.BETA, MU=model.MU)

    mags = [test_model.magnetization()]

    steps_taken = 0
    target_magnetization = (model.N_X * model.N_Y) * (1 - error_margin)
    while steps_taken < 500_000:
        test_model.run(resolution)
        steps_taken += resolution
        mags.append(test_model.magnetization())
        print(mags[-1], target_magnetization, steps_taken)

    print(np.linspace(0, steps_taken, len(mags)), mags)

    fig, ax = plt.subplots()

    ax.plot(np.linspace(0, steps_taken, len(mags)), mags, "o")
    ax.axhline(y=+model.N_X*model.N_Y, linestyle='--', color='black')
    ax.axhline(y=-model.N_X*model.N_Y, linestyle='--', color='black')

    ax.set_xlabel("Steps")
    ax.set_ylabel("Magnetization")

    ax.grid()
    ax.legend()

    # plt.show()
    fig.savefig(f"TestData/size={model.N_X*model.N_Y}_beta={model.BETA}.png", dpi=600)
    np.savetxt(f"TestData/size={model.N_X*model.N_Y}_beta={model.BETA}.txt", np.array(mags))

    return steps_taken


def test_energy():

    params = parse_input()

    N_X = int(params["N_X"])
    N_Y = int(params["N_Y"])
    MU = params["MU"]
    J = params["J"]
    BETA = params["BETA"]

    h = np.zeros((N_Y, N_X))
    h[0, :]      =  np.ones(h.shape[1])
    h[-1, :]     =  np.ones(h.shape[1])
    h[0, 40:60]  = -1
    h[-1, 40:60] = -1


    model = IsingModel(N_X, N_Y, J=J, BETA=BETA, MU=MU, h=h)

    last_e = model.hamiltonian()
    current_e = last_e
    print(last_e)
    print('-'*50)

    def manual_energy(spins):
        sum = 0
        for i, s in enumerate(spins):
            for j, ss in enumerate(s):
                if i!=0:
                    sum += ss*spins[i-1, j]
                if i!=N_Y-1:
                    sum += ss*spins[i+1, j]

                sum += ss*spins[i, (j-1)%N_X]
                sum += ss*spins[i, (j+1)%N_X]

        sum *= 1/2  # overcounting

        return -J * sum - MU * np.sum(h*spins)

    print(manual_energy(model.spins))


    for _ in range(100):
        ii, jj = np.random.randint(N_Y), np.random.randint(N_X)
        print(ii, jj)

        last_e = model.hamiltonian()
        e_diff = model.energy_diff(ii, jj)
        current_e += e_diff
        print(manual_energy(model.spins))
        print(current_e)

        model.spins[ii, jj] *= -1

        print(model.hamiltonian())
        print(model.hamiltonian() - last_e)
        print(e_diff)
        if np.abs(model.hamiltonian() - last_e - e_diff) > 0.1:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"*100)
        print('-'*50)


    print(J, MU)


def read_center_data(filename):
    data = []
    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i%4 == 0:
                line = line.split()
                line[1] = float(line[1])  # beta
                line[2] = int(line[2])    # N_Y
                new_entry = {"tag": line[0]}
                new_entry["beta"] = line[1]
                new_entry["N_Y"] = line[2]
            if i%4 == 1:
                new_entry["center_spins"] = float(line)
            if i%4 == 2:
                new_entry["center_line"] = float(line)
            if i%4 == 3:
                new_entry["center_column"] = float(line)
                data.append(new_entry)

    for d in data:
        print(d)
    print(pd.DataFrame(data=data))
    return


def read_avg_data(filename, line_nr=1):
    '''
    line_nr determines the metric:
        1 four/two center spins
        2 center line average
        3 center column average

    ignores tag
    '''
    data = {}
    with open(filename, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i%4 == 0:
                line = line.split()
                current_beta = float(line[1])
                current_ny = int(line[2])
            if i%4 == line_nr:
                if current_ny in data:
                    if current_beta in data[current_ny]:
                        data[current_ny][current_beta].append(float(line))
                    else:
                        data[current_ny][current_beta] = [float(line)]
                else:
                    data[current_ny] = {}
                    data[current_ny][current_beta] = [float(line)]

    for vv in data.values():
        for k, v in vv.items():
            vv[k] = np.mean(v)

    df = pd.DataFrame(data=data)
    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)

    return df


if __name__ == "__main__":
    for s in ["-", "+"]:
        for n in range(1, 4):
            print(s, n)
            df = read_avg_data(f"data_{s}.txt", line_nr=n)
            print(df)
            sns.heatmap(df, annot=True)
            plt.show()
