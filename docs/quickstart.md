The following code gives the minimal working example for this package:

```py
import slmcontrol
import numpy as np

#Initializes the SLM with the default display
slm = slmcontrol.SLMDisplay()

# Queries the SLM for its width and height
width, height = slm.width, slm.height

# Creates a grid of x and y coordinates
# Here, we are taking our units as pixels and the center of the SLM as (0,0)
# One could also use the physical dimensions of the SLM
x = np.linspace(-width/2, width/2, width)
y = np.linspace(-height/2, height/2, height)

# Calculates the field which we want to display
# In this case, we are using a Laguerre-Gaussian mode
desired = slmcontrol.lg(x, y, l=1, w = 200)

# The incoming field is assumed to be a larger gaussian beam
incoming = slmcontrol.lg(x, y, w = 500)

relative = desired / incoming

# We generate the hologram to be displayed on the SLM
holo = slmcontrol.generate_hologram(relative, 255, 50, 100)

# The hologram is then displayed on the SLM
slm.updateArray(holo)

# Finally, the SLM is closed
slm.close()
```

## Breakdown

Let's dissect this example, line by line:

```py
import slmcontrol
import numpy as np
```
This imports the `slmcontrol` package and NumPy.

```py
slm = slmcontrol.SLMDisplay()
```
This initializes the [`SLMDisplay`][src.slmcontrol.slm.SLMDisplay] object. By default, it uses the last available display for the Spatial Light Modulator (SLM). If you haven't connected your SLM and are just testing the package, this will be your primary display or a second monitor.

```py
width, height = slm.width, slm.height
```
This retrieves the width and height of the SLM in pixels. These dimensions will be used to create a coordinate grid.

```py
x = np.linspace(-width/2, width/2, width)
y = np.linspace(-height/2, height/2, height)
```
Creates a grid of x and y coordinates centered at (0,0). The coordinates span from negative half-width/height to positive half-width/height. This approach treats pixels as units, with the SLM center as the origin (0,0). Of course this choice is not unique, and one could also use the physical dimensions of the SLM instead.

```py
desired = slmcontrol.lg(x, y, l=1, w = 200)
```
Defines the desired output beam. In this case, it's a Laguerre-Gaussian mode with topological charge `l=1` and a beam waist of 200 pixels. The [`lg`][src.slmcontrol.structures.lg] function generates the complex field amplitude for this specific beam type.

```py
incoming = slmcontrol.lg(x, y, w = 500)

relative = desired / incoming
```
Sets up the incoming beam as a uniform plane wave and the relative field is calculated by dividing the desired field by the incoming field. This relative field represents the phase modulation needed to transform the incoming beam into the desired Laguerre-Gaussian mode.

```py
holo = slmcontrol.generate_hologram(relative, 255, 50, 100)
```
Generates the hologram that will be displayed on the SLM. This function calculates the phase pattern needed to transform the incoming plane wave into the desired Laguerre-Gaussian beam. The arguments specify:

- The relative field to be modulated.
- Maximum modulation value (255). This is the modulation that imparts a phase shift of 2π and should be set according to the SLM's specifications.
- The last two parameters are the period of the diffraction grating (in units of pixels) in the x (50px) and y (100px) directions. 

```py
slm.updateArray(holo)
```
Displays the calculated hologram on the SLM, which will modify the incoming light to produce the desired beam profile.

```py
slm.close()
```
Properly closes the SLM display, releasing any resources.

## Next Steps
Now that you have a basic understanding of how to use the `slmcontrol` package, you can explore more advanced features and functionalities. Here are some suggestions for what to do next:

- **Experiment with Different Beam Profiles**: Try generating different types of beams, such as Hermite-Gaussian ([`hg`][src.slmcontrol.structures.hg]) beams or other Laguerre-Gaussian modes, by modifying the parameters in the `lg` function. More spatial structures can be found in [here](reference.md#structures).

- **Get a high level view about the capabilities and design of the package in [Explanation][explanation]**