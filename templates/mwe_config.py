# Import everything from slmcontrol
from slmcontrol import *

# Initialize our slm display
slm = SLMdisplay(monitor=0)

# Path to the configuration file
config_path = 'templates/config.ini'
x, y = build_grid(config_path)

# The beam that we wish to produce.
# In this case, a Laguerre-Gaussian with p=0 and l=1
# We don't need the x,y, as it is calculated from the configuration file
desired = lg(x, y, 0, 1, .5)

# Generates the hologram that will be shown in the SLM using only the configuration file
# Notice that we don't even need to define the incoming beam as its wais is already defined in the configuration file
holo = generate_hologram(config_path, desired)

# Displays the generated hologram
slm.updateArray(holo)
