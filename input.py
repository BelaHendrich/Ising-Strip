import os
import numpy as np


OUT_DIR = "Data/"


def parse_input():
    for file in os.listdir():
        if file=="config.txt":
            return parse_config(file)
    raise FileNotFoundError("The configuration must be stored in " +
                            "a file named \"config.txt\"!")


def parse_config(filename):
    parameters = {"N_X": None,
                  "N_Y": None,
                  "MU": None,
                  "J": None,
                  "BETA": None,
                  }
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.split()
            if line[0] not in parameters.keys():
                raise ValueError(f"The parameters in {filename} " +
                                 f"must be in {list(parameters.keys())}!")
            parameters[line[0]] = float(line[2])

    for key, value in parameters.items():
        if value is None:
            raise ValueError(f"The value of {key} is missing!")

    return parameters


def parse_file(filename, as_dict=False):
    '''
    Read the model parameters from a simulation file in "Data/".
    If as_dict==True, all parameters are returned in a dictionary.
    Otherwise, N_X, N_Y and spins are unpacked and
    (N_X, N_Y, spins, {other_params}) is returned.
    '''
    filename = OUT_DIR + filename

    parameters = np.loadtxt(filename, dtype="str", max_rows=10, delimiter='=')
    parameters = dict(np.strings.strip(parameters))

    N_X = int(parameters["N_X"])
    N_Y = int(parameters["N_Y"])

    J    = float(parameters["J"])
    BETA = float(parameters["BETA"])
    MU   = float(parameters["MU"])

    fields = parameters["h"]
    h = [1 if d == "+" else -1 if d == "-" else 0 for d in fields]
    h = np.array(h)
    h.resize(N_Y, N_X)

    boundary_x = parameters["boundary_x"]
    boundary_y = parameters["boundary_y"]

    spins = parameters["spins"]
    current_state = [1 if d == "+" else -1 for d in spins]
    current_state = np.array(current_state)
    current_state.resize(N_Y, N_X)

    if as_dict:
        return {"N_X": N_X,
                "N_Y": N_Y,
                "J": J,
                "BETA": BETA,
                "MU": MU,
                "h": h,
                "spins": current_state,
                }

    return (N_X, N_Y, current_state,
            {"J": J,
             "BETA": BETA,
             "MU": MU,
             "h": h,
             })

