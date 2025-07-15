import gigildas.container as gc
from time import time
import numpy as np
import matplotlib.pyplot as plt

t = time()

container = gc.Container()
container.set_input("CB244_0LI.30m")
headers = container.get_headers()
intensity = container.get_data(headers)
intensity = np.mean(intensity, axis=0)
(spectro,) = container.get_sections(headers[0:1], gc.SpectroSection)
(calibration,) = container.get_sections(headers[0:1], gc.CalibrationSection)
positions = container.get_sections(headers, gc.PositionSection)

frequency = spectro.rest_frequency + (
        np.arange(1, spectro.channel_count + 1) - spectro.reference_channel) * spectro.frequency_resolution

freq = 90663.6
win_offset = 5
mask = (frequency <= freq + win_offset) & (frequency >= freq - win_offset)
plt.step(frequency, intensity, linewidth=.2)
plt.savefig("plot.svg")

print(f"time: {time() - t:.2f} s")


rng = np.random.default_rng()
x = rng.random(10) - 0.5
y = rng.random(10) - 0.5
z = np.hypot(x, y)
X = np.linspace(min(x), max(x))
Y = np.linspace(min(y), max(y))
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
interp = LinearNDInterpolator(list(zip(x, y)), z)
Z = interp(X, Y)
plt.pcolormesh(X, Y, Z, shading='auto')
plt.legend()
plt.colorbar()
plt.axis("equal")
plt.show()