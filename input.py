import os


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
