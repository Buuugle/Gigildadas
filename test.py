import gigildas.container as gc
from time import time
import numpy as np
import matplotlib.pyplot as plt

t = time()

container = gc.Container()
container.set_input("CB244_0LI.30m")
headers = container.get_headers()
intensity = container.get_data(headers)
(spectro,) = container.get_sections(headers[0:1], gc.SpectroSection)
frequency = spectro.rest_frequency + (
        np.arange(spectro.channel_count) - spectro.reference_channel + 1) * spectro.frequency_resolution
alpha = np.array([header.lambda_offset for header in headers]) * 180 * 3600 / np.pi
delta = np.array([header.beta_offset for header in headers]) * 180 * 3600 / np.pi
min_alpha, max_alpha = np.min(alpha), np.max(alpha)
min_delta, max_delta = np.min(delta), np.max(delta)

total_intensity = np.mean(intensity, axis=0)

lines_loop = total_intensity.copy()
lines_mask = np.zeros(len(lines_loop), dtype=bool)
lines_iterations = 2

for _ in range(lines_iterations):
    std = np.std(lines_loop)
    mean = np.mean(lines_loop)
    threshold_mask = lines_loop >= mean + 5 * std
    lines_mask |= threshold_mask
    lines_loop[threshold_mask] = mean

lines = np.where(lines_mask)[0]
breaks = np.where(np.diff(lines) > 1)[0] + 1
split_lines = np.split(lines, breaks)

plt.step(frequency, total_intensity, linewidth=.2)
plt.scatter(frequency[lines_mask], total_intensity[lines_mask], s=5, color="r")
plt.savefig("plot.svg")
plt.close()

win_size = 100
lines_set = set(lines)

for line in split_lines:
    center = max(line)
    print(center)
    begin = max(0, center - win_size)
    end = min(center + win_size, len(frequency))
    inter = slice(begin, end)

    win_freq = frequency[inter]
    win_int = intensity[:, inter].copy()

    clean_mask = np.array([i + begin not in lines_set for i in range(len(win_freq))])
    clean_freq = win_freq[clean_mask]
    clean_int = win_int[:, clean_mask]

    n = len(clean_freq)
    sum_x = np.sum(clean_freq)
    sum_xx = np.sum(clean_freq ** 2)
    sum_y = np.sum(clean_int, axis=1)
    sum_xy = np.sum(clean_int * clean_freq, axis=1)
    denominator = n * sum_xx - sum_x ** 2
    a = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y * sum_xx - sum_x * sum_xy) / denominator
    win_int -= a[:, np.newaxis] * win_freq + b[:, np.newaxis]

    total_int = np.mean(win_int[:, line - begin], axis=1)
    beam = 2_460_000 / frequency[center]
    sigma = beam / 3
    step = beam / 4
    gaussian = lambda x, y: np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    limit = beam ** 2
    num_alpha = int((max_alpha - min_alpha) / step) + 1
    num_delta = int((max_delta - min_delta) / step) + 1
    alpha_grid = np.linspace(min_alpha, max_alpha, num_alpha)
    delta_grid = np.linspace(min_delta, max_delta, num_delta)
    grid_x, grid_y = np.meshgrid(alpha_grid, delta_grid)
    grid_z = np.zeros_like(grid_x)

    for i in range(num_delta):
        for j in range(num_alpha):
            gx, gy = grid_x[i, j], grid_y[i, j]
            dist_mask = (alpha - gx) ** 2 + (delta - gy) ** 2 <= limit
            dist_alpha = alpha[dist_mask]
            dist_delta = delta[dist_mask]
            dist_int = total_int[dist_mask] * gaussian(dist_alpha - gx, dist_delta - gy)
            grid_z[i, j] = np.sum(dist_int)

    plt.gca().invert_xaxis()
    plt.pcolormesh(grid_x, grid_y, grid_z, shading="auto", cmap="gist_heat", vmin=0)
    plt.colorbar()
    plt.axis("equal")
    plt.savefig(f"{frequency[center]:.2f}.png")
    plt.close()

print(f"time: {time() - t:.2f} s")
