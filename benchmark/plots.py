import matplotlib.pyplot as plt
import numpy as np
import csv

def read_scaling_laws(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            backend = row['Backend']
            parameter = row['Parameter']
            values = eval(row['Values'])
            times = eval(row['Times'])
            data.append((backend, parameter, values, times))
    return data

def calculate_slope(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    slope, _ = np.polyfit(log_x, log_y, 1)
    return slope

data = read_scaling_laws('scaling_laws.csv')

for index, (backend, parameter, values, times) in enumerate(data):
    values = np.array(values)
    times = np.array(times)

    slope = calculate_slope(values, times)

    plt.figure()
    plt.loglog(values, times, marker='o', label=f'Slope: {slope:.2f}')
    plt.xlabel(parameter)
    plt.ylabel('Time')
    plt.title(f'Log-Log Plot for {parameter} ({backend})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(f'{backend}_{parameter}.png')

plt.show()