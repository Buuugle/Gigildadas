import gigildas.container as gc
from time import time
import numpy as np
import matplotlib.pyplot as plt

t = time()

container = gc.Container()
container.set_input("CB244_0LO.30m")
headers = container.get_headers()
intensity = container.get_data(headers)
(spectro,) = container.get_sections(headers[0:1], gc.SpectroSection)
frequency = spectro.rest_frequency + (
        np.arange(spectro.channel_count) - spectro.reference_channel + 1) * spectro.frequency_resolution

intensity = intensity[:, 310:]
frequency = frequency[310:]

total_intensity = np.mean(intensity, axis=0)
lines_loop = np.copy(total_intensity)
lines_mask = np.full((np.size(lines_loop)), False)
lines_iterations = 2
for _ in range(lines_iterations):
    std = np.std(lines_loop)
    mean = np.mean(lines_loop)
    lines_mask |= lines_loop >= mean + 5 * std
    lines_loop[lines_mask] = mean

lines = np.where(lines_mask)[0]
breaks = np.where(np.diff(lines) > 1)[0] + 1
split_lines = np.split(lines, breaks)

plt.step(frequency, total_intensity, linewidth=.2)
plt.scatter(frequency[lines_mask], total_intensity[lines_mask], s=5, color="r")
plt.savefig("plot.svg")
plt.close()

win_size = 400
print(lines)
for line in split_lines:
    print(frequency[line[0]])

    middle = line[len(line) // 2]
    begin = max(0, middle - win_size)
    end = min(middle + win_size, len(frequency))
    rang = slice(begin, end)
    line_frequency = frequency[rang]
    line_intensity = np.copy(intensity[:, rang])

    clean_lines = lines - begin
    clean_lines = clean_lines[(0 <= clean_lines) & (clean_lines < len(line_frequency))]
    clean_frequency = np.delete(line_frequency, clean_lines)
    clean_intensity = np.delete(line_intensity, clean_lines, axis=1)
    n = len(clean_frequency)
    sum_x = np.sum(clean_frequency)
    sum_xx = np.sum(clean_frequency ** 2)
    sum_y = np.sum(clean_intensity, axis=1)
    sum_xy = np.sum(clean_intensity * clean_frequency, axis=1)
    denominator = n * sum_xx - sum_x ** 2
    a = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y * sum_xx - sum_x * sum_xy) / denominator
    corrections = a[:, np.newaxis] * line_frequency + b[:, np.newaxis]

    line_intensity -= corrections

    total_line_intensity = np.mean(line_intensity, axis=0)
    plt.clf()
    plt.step(line_frequency, total_line_intensity, linewidth=.5)
    plt.savefig(f"{line[0]}.svg")
    plt.close()


print(f"time: {time() - t:.2f} s")