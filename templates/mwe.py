# Import everything from slmcontrol
from slmcontrol import *

# Initialize our slm display
slm = SLMdisplay(monitor=1)

# Constructs a grid that is defined by the size and resolution of the SLM
x, y = build_grid(15.36, 8.64, 1920, 1080)

# The input beam, which we assme to be a gaussian
incoming = hg(x, y, 0, 0, 3.3)

# The beam that we wish to produce.
# In this case, a Laguerre-Gaussian with p=0 and l=1
desired = lg(x, y, 0, 1, .5)

# Generates the hologram that will be shown in the SLM
holo = generate_hologram(desired, incoming, x, y, 96, 4, 5, 0, 0)

# Displays the generated hologram
slm.updateArray(holo)
