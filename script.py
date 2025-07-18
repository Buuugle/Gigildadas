import gigildadas.container as gc
import numpy as np
import matplotlib.pyplot as plt
from time import time

t = time()

# Use "gigildadas" lib to read the header, data and spectro section of all the observations
container = gc.Container()
window = "1LI"
container.set_input(f"CB244_{window}.30m")  # Set input file
print(f"Observation count: {container.get_size()}")
headers = container.get_headers()  # Get observations headers
print(f"Channel count: {headers[0].data_size}")
intensity = container.get_data(headers)  # Get intensity data (2D numpy array)
spectro = container.get_sections(headers[0:1], gc.SpectroSection)[0]  # Get spectro section of the first observation to compute frequency

# Compute frequency range + convert alpha and delta from radian to arcsec
frequency = spectro.rest_frequency + (
        np.arange(spectro.channel_count) - spectro.reference_channel + 1) * spectro.frequency_resolution
alpha = np.array([header.lambda_offset for header in headers]) * 180 * 3600 / np.pi
delta = np.array([header.beta_offset for header in headers]) * 180 * 3600 / np.pi
min_alpha, max_alpha = np.min(alpha), np.max(alpha)
min_delta, max_delta = np.min(delta), np.max(delta)

# Integrate intensity on all the surface
total_intensity = np.mean(intensity, axis=0)

# Detection of lines. Accuracy can be improved by computing std and mean around each channel instead of globally?
lines_mask = np.zeros(len(total_intensity), dtype=bool)
negative_mask = np.zeros(len(total_intensity), dtype=bool)
max_iterations = 10
std_threshold = 5
iterations = max_iterations
for i in range(max_iterations):
    intensity_masked = total_intensity[~(lines_mask | negative_mask)]
    std = np.std(intensity_masked)
    mean = np.mean(intensity_masked)
    new_mask = total_intensity >= mean + std_threshold * std
    if np.all(new_mask == lines_mask):
        iterations = i
        break
    lines_mask = new_mask
    negative_mask = total_intensity <= mean - std_threshold * std
print(f"Detection iteration count: {iterations}")
lines = np.where(lines_mask)[0]
split_lines = np.split(lines, np.where(np.diff(lines) > 1)[0] + 1)  # Group consecutive detected channels together to form lines
print(f"Detected line count: {len(split_lines)}")

# Plot total intensity in blue + detected line channels in red
plt.step(frequency, total_intensity, linewidth=.2)
plt.scatter(frequency[lines_mask], total_intensity[lines_mask], s=2, color="r")
plt.savefig(f"spectra/spectrum_{window}.svg")
plt.close()

# Compute map for each line
win_size = 100
lines_set = set(lines)
for line in split_lines:
    center = max(line)
    center_freq = frequency[center]
    print(f"Map: {center_freq:.2f} MHz")

    # Get win_size * 2 channels centered around line's frequency
    begin = max(0, center - win_size)
    end = min(center + win_size, len(frequency))
    inter = slice(begin, end)
    win_freq = frequency[inter]
    win_int = intensity[:, inter].copy()

    # Base line with linear fit
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

    # Gaussian convolution based on beam to grid data on 2D map (not perfect like original gildas)
    # See https://www.iram.fr/IRAMFR/GILDAS/doc/pdf/class.pdf
    # See https://git.iram.fr/gildas/gildas/-/blob/master/contrib/imager/lib/uvshort_lib.f90
    # See https://git.iram.fr/gildas/gildas/-/blob/master/contrib/imager/lib/util_uv.f90
    total_int = np.mean(win_int[:, line - begin], axis=1)
    beam = 2_460_000 / frequency[center]
    width = beam / 3
    step = beam / 4
    support = (5 * width) ** 2
    sigma = width / (2 * np.sqrt(np.log(2)))
    gaussian = lambda x, y: np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    num_alpha = int((max_alpha - min_alpha) / step) + 1
    num_delta = int((max_delta - min_delta) / step) + 1
    alpha_grid = np.linspace(min_alpha, max_alpha, num_alpha)
    delta_grid = np.linspace(min_delta, max_delta, num_delta)
    grid_x, grid_y = np.meshgrid(alpha_grid, delta_grid)
    grid_z = np.zeros_like(grid_x)
    for i in range(num_delta):  # Perhaps this loop can be vectorized with numpy for greater efficiency?
        for j in range(num_alpha):
            gx, gy = grid_x[i, j], grid_y[i, j]
            dist_mask = (alpha - gx) ** 2 + (delta - gy) ** 2 <= support
            dist_alpha = alpha[dist_mask]
            dist_delta = delta[dist_mask]
            dist_int = total_int[dist_mask] * gaussian(dist_alpha - gx, dist_delta - gy)
            grid_z[i, j] = np.sum(dist_int)

    # Plot maps with colors. Contours could be computed with RMS instead of automatically.
    plt.gca().invert_xaxis()
    plt.pcolormesh(grid_x, grid_y, grid_z, shading="auto", cmap="gist_heat", vmin=0)
    plt.colorbar()
    # plt.contour(grid_x, grid_y, grid_z, colors="white", linewidth=.75)
    plt.axis("equal")
    plt.savefig(f"maps/map_{window}_{center_freq:.2f}.png")
    plt.close()

print(f"Execution time: {time() - t:.2f} s")
