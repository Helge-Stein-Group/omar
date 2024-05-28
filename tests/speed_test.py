import platform
import timeit
from datetime import datetime

import numpy as np

n_samples = 10 ** 2
dim = 2
m_max = 10


def speed_test(command, output_file, setup, repeat=10, number=1):
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


def test_speed_full():
    speed_test(
        "regression.fit(x, y, m_max)",
        "../results/speeds_full.txt",
        "x, y, y_true = utils.generate_data(n_samples, dim)"
    )


def test_speed_update_cholesky():
    speed_test(
        "regression.update_cholesky(tri, vecs, vals)",
        "../results/speeds_cholesky.txt",
        "tri = np.triu(np.random.rand(100, 100))\n" +
        "vecs = [np.random.rand(100)]\n" +
        "vals = [2]",
        repeat=100
    )


def test_speed_basis():
    speed_test(
        "basis(vals)",
        "../results/speeds_basis.txt",
        "v = [0,1,2,3,4]\n" +
        "t = [0,0.5,-0.5,1,-1]\n" +
        "hinge = [True,False,True,False,True]\n" +
        "basis = regression.Basis(v, t, hinge)\n" +
        "vals = np.random.rand(100000, 10)",
        repeat=100
    )
