from utils import speed_test, monitor_scaling_laws


def test_speed_find_bases() -> None:
    speed_test(
        "import utils\n" +
        "import regression_noop\n" +
        "x, y, y_true = utils.generate_data(n_samples, dim)",
        "nbases, covariates, nodes, hinges, where, coefficients = regression_noop.find_bases(x, y)",
        "noop",
        "../results/speeds_find_bases.txt",
        repeat=100,
        number=1,
        n_samples=100,
        dim=2,
        m_max=10
    )


def test_monitor_scaling_laws() -> None:
    monitor_scaling_laws(
        "import utils\nimport regression_noop\n" +
        "x, y, y_true = utils.generate_data(n_samples, dim)",
        "nbases, covariates, nodes, hinges, where, coefficients = regression_noop.find_bases(x, y, max_nbases=m_max, max_ncandidates=m_max)",
        "noop_scaling_laws.png"
    )


def test_speed_update_cholesky() -> None:
    speed_test(
        "import utils\n" +
        "import regression_noop\n" +
        "import numpy as np\n" +
        "tri = np.triu(np.random.rand(100, 100))\n" +
        "vecs = [np.random.rand(100)]\n" +
        "vals = [2]",
        "regression_noop.update_cholesky(tri, vecs, vals)",
        "noop",
        "../results/speeds_update_cholesky.txt",
        repeat=100,
        number=1,
        n_samples=100,
        dim=2,
        m_max=10
    )


def test_speed_data_matrix() -> None:
    speed_test(
        "import utils\n" +
        "import regression_noop\n" +
        "x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(n_samples, dim)",
        "regression_noop.data_matrix(x, slice(nbases), covariates, nodes, hinges, where)",
        "noop",
        "../results/speeds_data_matrix.txt",
        repeat=100,
        number=1,
        n_samples=10 ** 5,
        dim=10,
        m_max=10
    )


def test_speed_update_init() -> None:
    speed_test(
        "import utils\n" +
        "import regression_noop\n" +
        "import numpy as np\n" +
        "x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(n_samples, dim)\n" +
        "old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]\n" +
        "nodes[2,2] = x[np.argmin(np.abs(x[:, 1] - 0.6)), 1]\n" +
        "parent_idx = 1\n" +
        "fit_matrix = regression_noop.calculate_fit_matrix(x, nbases, covariates, nodes, hinges, where)\n" +
        "candidate_mean = np.sum(fit_matrix, axis=0)[-1] / fit_matrix.shape[0]",
        "regression_noop.update_init(x, old_node, parent_idx, nbases, covariates, nodes, where, fit_matrix, candidate_mean)",
        "noop",
        "../results/speeds_update_init.txt",
        repeat=100,
        number=1,
        n_samples=10 ** 5,
        dim=10,
        m_max=10
    )


def test_speed_update_covariance_matrix() -> None:
    speed_test(
        "import utils\n" +
        "import regression_noop\n" +
        "import numpy as np\n" +
        "x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(n_samples, dim)\n" +
        "old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]\n" +
        "nodes[2,2] = x[np.argmin(np.abs(x[:, 1] - 0.6)), 1]\n" +
        "parent_idx = 1\n" +
        "fit_results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges, where, 3)\n" +
        "init_results = regression_noop.update_init(x, old_node, parent_idx, nbases, covariates, nodes, where, fit_results[2], fit_results[-2])\n" +
        "fit_results = list(fit_results)\n" +
        "fit_results[-2] = init_results[-1]\n" +
        "regression_noop.update_fit_matrix(fit_results[2], *init_results[:2])",
        "covariance_addition = regression_noop.update_covariance_matrix(fit_results[3], init_results[1], fit_results[2],init_results[0], fit_results[-3],init_results[-2], init_results[-1])",
        "noop",
        "../results/speeds_update_covariance_matrix.txt",
        repeat=100,
        number=1,
        n_samples=10 ** 5,
        dim=10,
        m_max=10
    )


def test_speed_decompose_addition() -> None:
    speed_test(
        "import regression_noop\n" +
        "import numpy as np\n" +
        "vec = np.arange(10)",
        "res = regression_noop.decompose_addition(vec)",
        "noop",
        "../results/speeds_decompose_addition.txt",
        repeat=100,
        number=1,
        n_samples=10 ** 5,
        dim=10,
        m_max=10
    )
