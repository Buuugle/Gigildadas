import gigildas.container as gc
from time import time

t = time()

container = gc.Container()
container.set_input("CB244_0LI.30m")
headers = container.get_headers()
data = container.get_data(headers)
sections = container.get_sections(headers, gc.GeneralSection)

print(time() - t)