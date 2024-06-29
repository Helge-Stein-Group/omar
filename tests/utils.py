import platform
import timeit
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import regression


def generate_data(n_samples: int, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.random.normal(2, 1, size=(n_samples, dim))
    zero = np.zeros(n_samples)
    y_true = (np.maximum(zero, (x[:, 0] - 1)) +
              np.maximum(zero, (x[:, 0] - 1)) * np.maximum(0, (x[:, 1] - 0.8)))
    y = y_true + 0.12 * np.random.normal(size=n_samples)
    return x, y, y_true


def data_generation_model(n_samples: int, dim: int) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, regression.OMARS]:
    x, y, y_true = generate_data(n_samples, dim)

    reference_model = regression.OMARS()
    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    reference_model.nbases = 3
    reference_model.covariates[1, 1] = 0
    reference_model.covariates[1:3, 2] = [0, 1]
    reference_model.nodes[1, 1] = x1
    reference_model.nodes[1:3, 2] = [x1, x08]
    reference_model.hinges[1, 1] = True
    reference_model.hinges[1:3, 2] = [True, True]
    reference_model.where[1, 1] = True
    reference_model.where[1:3, 2] = [True, True]
    reference_model.fit(x, y)
    return x, y, y_true, reference_model


def data_generation_model_noop(n_samples: int, dim: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, y_true = generate_data(n_samples, dim)

    covariates = np.zeros((11, 11), dtype=int)
    nodes = np.zeros((11, 11), dtype=float)
    hinges = np.zeros((11, 11), dtype=bool)
    where = np.zeros((11, 11), dtype=bool)

    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    covariates[1, 1] = 0
    covariates[1:3, 2] = [0, 1]
    nodes[1, 1] = x1
    nodes[1:3, 2] = [x1, x08]
    hinges[1, 1] = True
    hinges[1:3, 2] = [True, True]
    where[1, 1] = True
    where[1:3, 2] = [True, True]
    return x, y, y_true, 3, covariates, nodes, hinges, where


def speed_test(setup: str,
               command: str,
               method: str,
               output_file: str,
               repeat: int = 10,
               number: int = 1,
               n_samples=100,
               dim=2,
               m_max=10) -> None:
    setup = "import utils\nimport regression\n" + setup
    time = np.mean(
        timeit.repeat(command, setup=setup, globals={"n_samples": n_samples, "dim": dim, "m_max": m_max}, repeat=repeat,
                      number=number)
    )

    with open(output_file, "a") as result_file:
        result_file.write("{:<30} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {}\n".format(
            str(datetime.now()),
            "{:.6f}".format(time),
            method,
            n_samples,
            dim,
            m_max,
            platform.system()
        ))


def monitor_scaling_laws(setup: str, cmd: str, filename: str) -> None:
    time_over_nsamples = []
    time_over_dim = []
    time_over_mmax = []

    N = np.logspace(1, 4, 10, dtype=int)
    for n_samples in N:
        dim = 2
        m_max = 10
        variables = {
            "n_samples": n_samples,
            "dim": dim,
            "m_max": m_max
        }
        time = np.mean(
            timeit.repeat(cmd, setup=setup, globals=variables, repeat=1, number=1)
        )
        time_over_nsamples.append(time)
    M = np.linspace(5, 55, 50, dtype=int)
    for m_max in M:
        n_samples = 10 ** 2
        dim = 2
        variables = {
            "n_samples": n_samples,
            "dim": dim,
            "m_max": m_max
        }
        time = np.mean(
            timeit.repeat(cmd, setup=setup, globals=variables, repeat=1, number=1)
        )
        time_over_mmax.append(time)
    d = np.linspace(2, 12, 10, dtype=int)
    for dim in d:
        n_samples = 10 ** 2
        m_max = 10
        variables = {
            "n_samples": n_samples,
            "dim": dim,
            "m_max": m_max
        }
        time = np.mean(
            timeit.repeat(cmd, setup=setup, globals=variables, repeat=1, number=1)
        )
        time_over_dim.append(time)

    fig, ax = plt.subplots(3, 1, figsize=(10, 5))

    scaling_N = np.polyfit(np.log(N), np.log(time_over_nsamples), 1)[0]
    ax[0].plot(N, time_over_nsamples, label=f"Time over n_samples O(N^{scaling_N})", color="blue")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("n_samples")
    ax[0].set_ylabel("Time (s)")
    ax[0].legend()

    scaling_M = np.polyfit(np.log(M), np.log(time_over_mmax), 1)[0]
    ax[1].plot(M, time_over_mmax, label=f"Time over m_max O(M^{scaling_M})", color="red")
    ax[1].set_xlabel("m_max")
    ax[1].set_ylabel("Time (s)")
    ax[1].legend()

    scaling_d = np.polyfit(np.log(d), np.log(time_over_dim), 1)[0]
    ax[2].plot(d, time_over_dim, label=f"Time over dim O(d^{scaling_d})", color="green")
    ax[2].set_xlabel("dim")
    ax[2].set_ylabel("Time (s)")
    ax[2].legend()

    plt.show()
    plt.savefig("../results/" + filename)
    plt.close()
