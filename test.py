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

frequency = spectro.rest_frequency + (
        np.arange(spectro.channel_count) - spectro.reference_channel + 1) * spectro.frequency_resolution

freq = 90663.6
win_offset = 10
mask = (frequency <= freq + win_offset) & (frequency >= freq - win_offset)
plt.step(frequency[mask], intensity[mask], linewidth=.2)
plt.savefig("plot.svg")

print(f"time: {time() - t:.2f} s")
