import regression
import utils
import platform
import timeit
from datetime import datetime
import numpy as np

n_samples = 10 ** 2
dim = 2
m_max = 10

x, y, y_true = utils.generate_data(n_samples, dim)

time = np.mean(timeit.repeat("regression.fit(x, y, m_max)", globals=globals(), repeat=10, number=1))
#time = np.mean(timeit.repeat("regression.update_cholesky(np.triu(np.random.rand(100, 100)), [np.random.rand(100)], [2])", globals=globals(), repeat=1000, number=1))

with open("speeds.txt", "a") as result_file:
    result_file.write("{:<30} | {:<10} | {:<10} | {:<10} | {:<10} | {}\n".format(
        str(datetime.now()),
        "{:.6f}".format(time),
        str(n_samples),
        str(dim),
        str(m_max),
        platform.system()
    ))
