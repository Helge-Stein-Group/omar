import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from scipy.linalg import cho_factor, cho_solve, qr, solve_triangular, qr_insert, qr_update


def lstsq(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def cho(a, b):
    square = a.T @ (a - np.mean(a, axis=0))
    square += np.eye(square.shape[0]) * 1e-6
    rhs = a.T @ (b - np.mean(b))
    return cho_solve(cho_factor(square), rhs)


def qr_eye(a, b):
    a += np.eye(*a.shape) * 1e-6
    q, r = qr(a, mode='economic')
    return solve_triangular(r, q.T @ b)


def qr_trim(a, b):
    q, r = qr(a, mode='economic')
    dependent_cols = np.where(~r.any(axis=0))[0]
    r = np.delete(r, dependent_cols, axis=0)
    r = np.delete(r, dependent_cols, axis=1)
    q = np.delete(q, dependent_cols, axis=1)
    coef = solve_triangular(r, q.T @ b)
    coef = np.append(coef, np.zeros(len(dependent_cols)))
    return coef


def qr_sq(a, b):
    square = a.T @ (a - np.mean(a, axis=0))
    square += np.eye(square.shape[0]) * 1e-6
    rhs = a.T @ (b - np.mean(b))
    q, r = qr(square)
    return solve_triangular(r, q.T @ rhs)


def qr_insert_raw(a_add, b, q, r):
    q, r = qr_insert(q, r, a_add, q.shape[1], "col")
    dependent_cols = np.where(~r.any(axis=0))[0]
    r = np.delete(r, dependent_cols, axis=0)
    r = np.delete(r, dependent_cols, axis=1)
    q = np.delete(q, dependent_cols, axis=1)
    coef = solve_triangular(r, q.T @ b)
    coef = np.append(coef, np.zeros(len(dependent_cols)))
    return coef


def qr_insert_sq(a, a_add, b, q, r, b_og):
    vec_add = a.T @ a_add
    q, r = qr_insert(q, r, vec_add, q.shape[1], "col")
    vec_add = np.append(vec_add, np.inner(a_add, a_add) + 1e-6)
    q, r = qr_insert(q, r, vec_add, q.shape[0], "row")

    b = np.append(b, np.inner(a_add, b_og))

    return solve_triangular(r, q.T @ b)


times = {
    "lstsq": [],
    "cho": [],
    "qr_eye": [],
    "qr_trim": [],
    "qr_sq": [],
    "qr_insert_raw": [],
    "qr_insert_sq": [],
}

labels = np.linspace(10, 1e5, 100, dtype=int)
for n in labels:
    a = np.random.rand(n, 10)
    b = np.random.rand(n)

    for method in times.keys():
        func_string = f"{method}(a, b)"
        if method == "qr_insert_raw":
            a_trim = a[:, :-1]
            q, r = qr(a_trim, mode='economic')
            func_string = f"{method}(a[:, -1], b, q, r)"
        elif method == "qr_insert_sq":
            a_trim = a[:, :-1]
            rhs = a_trim.T @ b
            square = a_trim.T @ (a_trim - np.mean(a_trim, axis=0))
            square += np.eye(square.shape[0]) * 1e-6
            q, r = qr(square)
            func_string = f"{method}(a_trim, a[:, -1], rhs, q, r, b)"

        times[method].append(timeit(func_string, globals=globals(), number=10))

for method, time in times.items():
    plt.plot(labels, time, label=method)

plt.legend()
plt.show()
