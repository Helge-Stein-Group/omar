from utils import speed_test, monitor_scaling_laws


def test_speed_find_bases() -> None:
    speed_test(
        "import utils\n" +
        "import regression\n" +
        "x, y, y_true = utils.generate_data(n_samples, dim)\n" +
        "model = regression.OMARS()",
        "model.find_bases(x, y)",
        "oop",
        "../results/speeds_find_bases.txt",
        repeat=100,
        number=1,
        n_samples=100,
        dim=2,
        m_max=10
    )


def test_monitor_scaling_laws() -> None:
    monitor_scaling_laws(
        "import utils\nimport regression\n" +
        "x, y, y_true = utils.generate_data(n_samples, dim)\n" +
        "model = regression.OMARS()",
        "model.find_bases(x, y)",
        "oop_scaling_laws.png"
    )


def test_speed_update_cholesky() -> None:
    speed_test(
        "import utils\n" +
        "import regression\n" +
        "import numpy as np\n" +
        "tri = np.triu(np.random.rand(100, 100))\n" +
        "vecs = [np.random.rand(100)]\n" +
        "vals = [2]",
        "regression.update_cholesky(tri, vecs, vals)",
        "oop",
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
        "import regression\n" +
        "x, y, y_true, model = utils.data_generation_model(n_samples, dim)",
        "model.data_matrix(x, slice(model.nbases))",
        "oop",
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
        "import regression\n" +
        "import numpy as np\n" +
        "x, y, y_true, model = utils.data_generation_model(n_samples, dim)\n" +
        "old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]\n" +
        "model.nodes[2,2] = x[np.argmin(np.abs(x[:, 1] - 0.6)), 1]\n" +
        "parent_idx = 1",
        "model.update_init(x, old_node, parent_idx)",
        "oop",
        "../results/speeds_update_init.txt",
        repeat=100,
        number=1,
        n_samples=10 ** 5,
        dim=10,
        m_max=10
    )


def test_speed_covariance_update() -> None:
    speed_test(
        "import utils\n" +
        "import regression\n" +
        "import numpy as np\n" +
        "x, y, y_true, model = utils.data_generation_model(n_samples, dim)\n" +
        "old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]\n" +
        "model.nodes[2,2] = x[np.argmin(np.abs(x[:, 1] - 0.6)), 1]\n" +
        "parent_idx = 1\n" +
        "model.update_init(x, old_node, parent_idx)",
        "model.update_covariance_matrix()",
        "oop",
        "../results/speeds_update_covariance_matrix.txt",
        repeat=100,
        number=1,
        n_samples=10 ** 5,
        dim=10,
        m_max=10
    )


def test_speed_decompose_addition() -> None:
    speed_test(
        "import utils\n" +
        "import regression\n" +
        "import numpy as np\n" +
        "x, y, y_true, model = utils.data_generation_model(n_samples, dim)\n" +
        "model.covariance_matrix = np.zeros((10, 10))\n" +
        "vec = np.arange(10)",
        "model.decompose_addition(vec)",
        "oop",
        "../results/speeds_decompose_addition.txt",
        repeat=100,
        number=1,
        n_samples=10 ** 5,
        dim=10,
        m_max=10
    )
