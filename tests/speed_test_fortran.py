from utils import speed_test

def test_speed_find_bases() -> None:
    speed_test(
        "import utils\n" +
        "import fortran.omars as fortran\n" +
        "x, y, y_true = utils.generate_data(n_samples, dim)",
        "nbases, covariates, nodes, hinges, where, coefficients = fortran.omars.find_bases(x, y, y.mean(), 11, 5, 0, 3)",
        "fortran",
        "../results/speeds_find_bases.txt",
        repeat=100,
        number=1,
        n_samples=100,
        dim=2,
        m_max=10
    )