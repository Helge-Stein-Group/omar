import platform
import timeit
from datetime import datetime

import numpy as np

n_samples = 10 ** 2
dim = 2
m_max = 10


def speed_test(command: str, output_file: str, setup: str, repeat: int = 10, number: int = 1) -> None:
    setup = "import utils\nimport regression\n" + setup
    time = np.mean(
        timeit.repeat(command, setup=setup, globals=globals(), repeat=repeat,
                      number=number)
    )

    with open(output_file, "a") as result_file:
        result_file.write("{:<30} | {:<10} | {:<10} | {:<10} | {:<10} | {}\n".format(
            str(datetime.now()),
            "{:.6f}".format(time),
            n_samples,
            dim,
            m_max,
            platform.system()
        ))


def test_speed_find_bases() -> None:
    speed_test(
        "model.find_bases(x, y)",
        "../results/speeds_full.txt",
        "x, y, y_true = utils.generate_data(n_samples, dim)\n" +
        "model = regression.OMARS()",
    )


def test_speed_update_cholesky() -> None:
    speed_test(
        "regression.update_cholesky(tri, vecs, vals)",
        "../results/speeds_cholesky.txt",
        "tri = np.triu(np.random.rand(100, 100))\n" +
        "vecs = [np.random.rand(100)]\n" +
        "vals = [2]",
        repeat=100
    )


def test_speed_basis() -> None:
    global n_samples
    n_samples = 10 ** 5
    global dim
    dim = 10
    speed_test(
        "model(x)",
        "../results/speeds_basis.txt",
        f"x, y, y_true, model = utils.data_generation_model({n_samples}, {dim})",
        repeat=100
    )


def test_speed_update_init() -> None:
    global n_samples
    n_samples = 10 ** 5
    global dim
    dim = 10
    speed_test(
        "model.update_init(x, old_node, parent_idx)",
        "../results/speeds_update_init.txt",
        "x, y, y_true, model = utils.data_generation_model(n_samples, dim)\n" +
        "old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]\n" +
        "model.nodes[2,2] = x[np.argmin(np.abs(x[:, 1] - 0.6)), 1]\n" +
        "parent_idx = 1",
        repeat=100
    )


def test_speed_covariance_update() -> None:
    global n_samples
    n_samples = 10 ** 5
    global dim
    dim = 10
    speed_test(
        "model.update_covariance_matrix()",
        "../results/speeds_covariance_update.txt",
        "x, y, y_true, model = utils.data_generation_model(n_samples, dim)\n" +
        "old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]\n" +
        "model.nodes[2,2] = x[np.argmin(np.abs(x[:, 1] - 0.6)), 1]\n" +
        "parent_idx = 1\n" +
        "model.update_init(x, old_node, parent_idx)",
        repeat=100
    )


def test_speed_decompose_addition() -> None:
    speed_test(
        "model.decompose_addition(vec)",
        "../results/speeds_decompose_addition.txt",
        "x, y, y_true, model = utils.data_generation_model(n_samples, dim)\n" +
        "model.covariance_matrix = np.zeros((10, 10))\n" +
        "vec = np.arange(10)",
        repeat=100
    )
