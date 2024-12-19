import timeit
import platform

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from jaxtyping import Float

from omars import OMARS

N_SAMPLES = 100
DIM = 2


def generate_data(n_samples: int = N_SAMPLES, dim: int = DIM) \
        -> tuple[
            Float[np.ndarray, "{n_samples} {dim}"],
            Float[np.ndarray, "{n_samples}"],
            Float[np.ndarray, "{n_samples}"]
        ]:
    x = np.random.normal(2, 1, size=(n_samples, dim))
    y_true = (x[:, 0] +
              np.maximum(0, (x[:, 0] - 1)) +
              np.maximum(0, (x[:, 0] - 1)) * x[:, 1]+
              np.maximum(0, (x[:, 0] - 1)) * np.maximum(0, (x[:, 1] - 0.8)))
    y = y_true + 0.12 * np.random.normal(size=n_samples)
    return x, y, y_true


def reference_model(x: Float[np.ndarray, "n_samples"]) -> OMARS:
    model = OMARS()

    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]

    model.nbases = 5
    model.mask[1, 1:3] = True
    model.mask[1:3, 3:5] = [True, True]
    model.truncated[1, 2:5] = True
    model.truncated[2, 4] = True
    model.cov[1, 1:5] = 0
    model.cov[2, 3:5] = 1
    model.root[1, 2:5] = x1
    model.root[2, 4] = x08

    return model


def reference_data_matrix(x: Float[np.ndarray, "n_samples dim"]) \
        -> tuple[Float[np.ndarray, "{n_samples} 4"], Float[np.ndarray, "4"]]:
    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    ref_data_matrix = np.column_stack([
        x[:, 0],
        np.maximum(0, x[:, 0] - x1),
        np.maximum(0, x[:, 0] - x1) * x[:, 1],
        np.maximum(0, x[:, 0] - x1) * np.maximum(0, x[:, 1] - x08),
    ])

    ref_data_matrix_mean = ref_data_matrix.mean(axis=0)
    ref_data_matrix -= ref_data_matrix_mean

    return ref_data_matrix, ref_data_matrix_mean


def reference_covariance_matrix(ref_data_matrix: Float[np.ndarray, "n_samples 4"]) -> Float[np.ndarray, "4 4"]:
    ref_cov_matrix = ref_data_matrix.T @ ref_data_matrix + np.eye(ref_data_matrix.shape[1]) * 1e-8

    return ref_cov_matrix


def reference_rhs(y: Float[np.ndarray, "n_samples"], ref_data_matrix: Float[np.ndarray, "n_samples 4"]) -> Float[
    np.ndarray, "n_samples"]:
    ref_rhs = ref_data_matrix.T @ (y - y.mean())

    return ref_rhs


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

    N = np.logspace(1, 4, 20, dtype=int)
    for n_samples in N:
        dim = 2
        m_max = 11
        variables = {
            "n_samples": n_samples,
            "dim": dim,
            "m_max": m_max
        }
        time = np.mean(
            timeit.repeat(cmd, setup=setup, globals=variables, repeat=1, number=1)
        )
        time_over_nsamples.append(time)
    M = np.linspace(3, 17, 8, dtype=int)
    for m_max in M:
        n_samples = 10 ** 3
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
    d = np.linspace(2, 12, 11, dtype=int)
    for dim in d:
        n_samples = 10 ** 3
        m_max = 11
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
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("Time (s)")
    # ax[0].legend()

    scaling_M = np.polyfit(np.log(M), np.log(time_over_mmax), 1)[0]
    ax[1].plot(M, time_over_mmax, label=f"Time over m_max O(M^{scaling_M})", color="red")
    ax[1].set_xlabel("M_max")
    ax[1].set_ylabel("Time (s)")
    # ax[1].legend()

    scaling_d = np.polyfit(np.log(d), np.log(time_over_dim), 1)[0]
    ax[2].plot(d, time_over_dim, label=f"Time over dim O(d^{scaling_d})", color="green")
    ax[2].set_xlabel("d")
    ax[2].set_ylabel("Time (s)")
    # ax[2].legend()

    plt.savefig("../results/" + filename + ".png")

    np.savetxt("../results/" + filename + "N.csv", np.array([N, time_over_nsamples]), delimiter=",")
    np.savetxt("../results/" + filename + "M_max.csv", np.array([M, time_over_mmax]), delimiter=",")
    np.savetxt("../results/" + filename + "d.csv", np.array([d, time_over_dim]), delimiter=",")
