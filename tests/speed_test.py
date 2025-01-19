import csv
import timeit

from datetime import datetime

import numpy as np

import omar

def test_speed():
    setup = r"""
import omar
import tests.utils as utils
            
x, y, y_true = utils.generate_data(10000, 10)
model = omar.OMAR(backend=backend)
"""
    command = "model.find_bases(x, y)"
    results = []

    for backend in omar.Backend:
        time = np.mean(timeit.repeat(command, setup=setup, globals={"backend": backend}, repeat=10, number=1))
        results.append([str(datetime.now()), "{:.6f}".format(time), backend])


    with open("../benchmark/speeds_find_bases.csv", "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Time", "Backend"])
        writer.writerows(results)

def test_scaling_laws() -> None:
    setup = r"""
import omar
import tests.utils as utils

x, y, y_true = utils.generate_data(n_samples, dim)
model = omar.OMAR(max_nbases=max_nbases, max_ncandidates=5, backend=backend)
"""

    command = "model.find_bases(x, y)"

    variables = {
        "backend": None,
        "n_samples": 10 ** 4,
        "dim": 4,
        "max_nbases": 11
    }

    results = {}
    for backend in omar.Backend:
        results[backend] = {}
        variables["backend"] = backend

        N = np.logspace(1, 4, 20, dtype=int)
        N_times = []
        for n_samples in N:
            variables["n_samples"] = n_samples
            N_times.append(np.mean(timeit.repeat(command, setup=setup, globals=variables, repeat=10, number=1)))

        results[backend]["n_samples"] = (N, N_times)

        M = np.linspace(3, 17, 8, dtype=int)
        M_times = []
        for m_max in M:
            variables["max_nbases"] = m_max
            M_times.append(np.mean(timeit.repeat(command, setup=setup, globals=variables, repeat=10, number=1)))
        results[backend]["max_nbases"] = (M, M_times)
        variables["max_nbases"] = 11

        d = np.linspace(2, 10, 5, dtype=int)
        d_times = []
        for dim in d:
            variables["dim"] = dim
            d_times.append(np.mean(timeit.repeat(command, setup=setup, globals=variables, repeat=10, number=1)))
        results[backend]["dim"] = (d, d_times)

    with open("../benchmark/scaling_laws.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Backend", "Parameter", "Values", "Times"])
        for backend, params in results.items():
            for param, (values, times) in params.items():
                writer.writerow([backend, param, list(values), list(times)])