import csv
import timeit

from datetime import datetime

import numpy as np

import omars

def test_speed():
    setup = r"""
import omars
import tests.utils as utils
            
x, y, y_true = utils.generate_data(10000, 10)
model = omars.OMARS(backend=backend)
"""
    command = "model.find_bases(x, y)"
    results = []

    for backend in omars.Backend:
        time = np.mean(timeit.repeat(command, setup=setup, globals={"backend": backend}, repeat=10, number=1))
        results.append([str(datetime.now()), "{:.6f}".format(time), backend])


    with open("../benchmark/speeds_find_bases.csv", "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Time", "Backend"])
        writer.writerows(results)

def test_scaling_laws() -> None:
    setup = r"""
import omars
import tests.utils as utils

x, y, y_true = utils.generate_data(n_samples, dim)
model = omars.OMARS(max_nbases=max_nbases, max_ncandidates=5, backend=backend)
"""

    command = "model.find_bases(x, y)"

    variables = {
        "backend": None,
        "n_samples": 10 ** 4,
        "dim": 4,
        "max_nbases": 11
    }

    results = {}
    for backend in omars.Backend:
        results[backend] = {}
        variables["backend"] = backend

        N = np.logspace(1, 4, 20, dtype=int)
        N_times = []
        for n_samples in N:
            variables["n_samples"] = n_samples
            N_times += [np.mean(timeit.repeat(command, setup=setup, globals=variables, repeat=10, number=1))]

        results[backend]["n_samples"] = np.array([[N], [N_times]])

        M = np.linspace(3, 17, 8, dtype=int)
        M_times = []
        for m_max in M:
            variables["max_nbases"] = m_max
            M_times += [np.mean(timeit.repeat(command, setup=setup, globals=variables, repeat=10, number=1))]
        results[backend]["max_nbases"] = np.array([[M], [M_times]])
        variables["max_nbases"] = 11

        d = np.linspace(2, 10, 5, dtype=int)
        d_times = []
        for dim in d:
            variables["dim"] = dim
            d_times += [np.mean(timeit.repeat(command, setup=setup, globals=variables, repeat=10, number=1))]
        results[backend]["dim"] = np.array([[d], [d_times]])

    with open("../benchmark/scaling_laws.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Backend", "Parameter", "Values", "Times"])
        for backend, params in results.items():
            for param, data in params.items():
                values, times = data
                writer.writerow([backend, param, values.tolist(), times.tolist()])