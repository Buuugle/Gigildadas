import gigildas.container as gc
from time import time
import numpy as np

t = time()

container = gc.Container()
container.set_input("CB244_0LI.30m")
headers = container.get_headers()
data = container.get_data(headers)
data = np.mean(data, axis=0)
(spectro,) = container.get_sections(headers[0:1], gc.SpectroSection)
print(spectro.rest_frequency, spectro.reference_channel)

print(time() - t)
