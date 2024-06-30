from utils import speed_test


def test_speed_data_matrix() -> None:
    speed_test(
        "import utils\n" +
        "import fortran.data_matrix\n" +
        "x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(n_samples, dim)",
        "fortran.data_matrix.omars_data_matrix(x, 0, nbases, covariates, nodes, hinges, where)",
        "fortran",
        "../results/speeds_data_matrix.txt",
        repeat=100,
        number=1,
        n_samples=10 ** 5,
        dim=10,
        m_max=10
    )
