from utils import speed_test, monitor_scaling_laws


def test_speed_find_bases() -> None:
    speed_test(
        "import utils\n" +
        "import fortran.omars as fortran\n" +
        "x, y, y_true = utils.generate_data(n_samples, dim)",
        "nbases, covariates, nodes, hinges, where, coefficients = fortran.omars.find_bases(x, y, y.mean(), m_max, m_max, 0, 3)",
        "fortran",
        "../results/speeds_find_bases.txt",
        repeat=10,
        number=1,
        n_samples=10**4,
        dim=10,
        m_max=11
    )

def test_monitor_scaling_laws() -> None:
    monitor_scaling_laws(
        "import utils\n" +
        "import fortran.omars as fortran\n" +
        "x, y, y_true = utils.generate_data(n_samples, dim)",
        "nbases, covariates, nodes, hinges, where, coefficients = fortran.omars.find_bases(x, y, y.mean(), m_max, m_max, 0, 3)",
        "fortran"
    )