import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import cho_factor
from regression import update_cholesky

# Initialize list to store matrix sizes and corresponding times
sizes = []
times = []
times_update = []
# Generate matrices of increasing size
for n in np.logspace(2, 4, num=10, dtype=int):
    # Generate a random symmetric, positive-definite matrix
    A = np.random.rand(n, n)
    A = A @ A.T

    # Measure the time taken to compute the Cholesky decomposition
    start_time = time.time()
    tri, lower = cho_factor(A)
    end_time = time.time()

    vec = [np.random.rand(n)]
    val = [1]
    start_time_update = time.time()
    update_cholesky(tri, vec, val)
    end_time_update = time.time()

    # Store the matrix size and time taken
    sizes.append(n)
    times.append(end_time - start_time)
    times_update.append(end_time_update - start_time_update)

# Convert lists to numpy arrays
sizes = np.array(sizes)
times = np.array(times)
times_update = np.array(times_update)

# Fit a cubic polynomial to the log-transformed sizes and times
coeffs3 = np.polyfit(sizes + 1e-10, times + 1e-10, 3)
coeffs2 = np.polyfit(sizes + 1e-10, times + 1e-10, 2)

coeffs3_update = np.polyfit(sizes + 1e-10, times_update + 1e-10, 3)
coeffs2_update = np.polyfit(sizes + 1e-10, times_update + 1e-10, 2)

# Create a range of sizes for plotting the fitted polynomial
sizes_fit = np.linspace(min(sizes), max(sizes), 500)

# Compute the fitted times
times_fit3 = np.polyval(coeffs3, sizes_fit)
times_fit2 = np.polyval(coeffs2, sizes_fit)

times_fit3_update = np.polyval(coeffs3_update, sizes_fit)
times_fit2_update = np.polyval(coeffs2_update, sizes_fit)

mse3 = np.mean((times - np.polyval(coeffs3, sizes))**2)
mse2 = np.mean((times - np.polyval(coeffs2, sizes))**2)

mse3_update = np.mean((times_update - np.polyval(coeffs3_update, sizes))**2)
mse2_update = np.mean((times_update - np.polyval(coeffs2_update, sizes))**2)

# Plot the actual times and the fitted polynomial
fig, ax = plt.subplots(2, 1, figsize=(10, 12))
ax[0].scatter(sizes, times, label="Actual times")
ax[0].plot(sizes_fit, times_fit3, label=f"Fit (cubic), MSE={mse3:.2e}")
ax[0].plot(sizes_fit, times_fit2, label=f"Fit (quadratic), MSE={mse2:.2e}")
ax[0].legend()
ax[0].set_xlabel("Matrix size")
ax[0].set_ylabel("Time (s)")
ax[0].set_title("Cholesky decomposition time complexity")

ax[1].scatter(sizes, times_update, label="Actual times")
ax[1].plot(sizes_fit, times_fit3_update, label=f"Fit (cubic), MSE={mse3_update:.2e}, cubic coefficient={coeffs3_update[0]:.2e}")
ax[1].plot(sizes_fit, times_fit2_update, label=f"Fit (quadratic), MSE={mse2_update:.2e}")
ax[1].legend()
ax[1].set_xlabel("Matrix size")
ax[1].set_ylabel("Time (s)")
ax[1].set_title("Cholesky update time complexity")

plt.tight_layout()
plt.show()
