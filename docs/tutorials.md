## Quick Start

The following code gives the minimal working example for this package:


```py
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
```

### Explanation

Let us dissect this example, line by line:

```py
from slmcontrol import *
```
This is straightforward. We just import everything from `slmcontrol`.

```py
slm = SLMdisplay(monitor=1)
```
This initializes our [`SLMdisplay`][src.slmcontrol.slm.SLMdisplay] object. The argument `monitor` specifies which monitor will display the masks.
`monitor=0` corresponds to the primary monitor, and the subsequent monitors are associated with numbers `1,2,3,...`. If you still haven't connected your SLM, and have only one monitor (for example, you are just testing this package), you must change this line to `slm = SLMdisplay(monitor=1)`.

```py
x, y = build_grid(15.36, 8.64, 1920, 1080)
```
This line uses the [`build_grid`][src.slmcontrol.hologram.build_grid] function to construct the grid that will be used in the calculations. The arguments are the width, the height, and the resolution of the SLM. The precise units don't matter, as long as one sticks with the same choice throughout the code. Here, we chose millimeters.

```py
incoming = hg(x, y, 0, 0, 3.3)
```
Here we define the beam that arrives at the SLM. We assume that it is a Gaussian (Hermite-Gaussian [`hg`][src.slmcontrol.structures.hg] with both indices set to `0`) with a waist of $3.3$mm, and that it is centered at the SLM. If the incoming beam is **much** larger than the beam that you wish to produce, one may set this to a plane wave `incoming = np.ones((1080,1920))`.

```py
desired = lg(x, y, 0, 1, .5)
```
Now we define the beam that we wish to produce. It is a Laguerre-Gaussian with radial index `p=0`, topological charge `l=1` and a waist of $0.5$mm. This is just a numpy array of complex numbers, representing a field amplitude at a fixed plane. There are a bunch of predefined functions that calculate transverse structures (see [Structures](reference.md/#structures)), but this can be any "honest" field. Feel free to implement your own.

```py
holo = generate_hologram(desired, incoming, x, y, 96, 4, 5, 0, 0)
```

Now we use the [`generate_hologram`][src.slmcontrol.hologram.generate_hologram] function in order to calculate the hologram that needs to be shown in order to produce our desired output. This is a grayscale image (here, represented as a 2D numpy array of 8-bit unsigned integers). The parameter `max_modulation=96` is the maximum value in this array, and corresponds to a phase modulation of $2\pi$. You have to check the manual of the SLM in order to find this correct value. Note that it is wavelength dependent. We also specify the period of our diffraction grating: `4` pixels in the x direction, and `5` pixels in the y direction. If you want to understand the math behind these holograms, check the suggested references at the end of this section.

```python
slm.updateArray(holo)
```
This simply displays the calculated hologram to the SLM.

!!! References
    [1] Victor Arrizón, Ulises Ruiz, Rosibel Carrada, and Luis A. González, 
        "Pixelated phase computer holograms for the accurate encoding of scalar complex fields," 
        J. Opt. Soc. Am. A 24, 3500-3507 (2007)

    [2] Thomas W. Clark, Rachel F. Offer, Sonja Franke-Arnold, Aidan S. Arnold, and Neal Radwell, 
        "Comparison of beam generation techniques using a phase only spatial light modulator," 
        Opt. Express 24, 6249-6264 (2016)

## Configuration file

When working with an SLM, there are a lot of adjustable parameters that, once you have the experimental setup ready, will remain fixed. Examples are the SLM size, resolution, diffraction grating periods, and so on. For that reason, it may become tedious to write these parameters every time you write a new program. Therefore, we developed a system in which the parameters are written on a file and then most of the functions have a version that simply reads this file. First, we show a template of the file, which we will call `config.ini`:

```ini
[slm]
width = 15.36
height = 8.64
resX = 1920
resY = 1080
max_modulation = 96

[incoming]
waist = 1
xoffset = 0
yoffset = 0

[grating]
xperiod = 4
yperiod = 5
```

This is a `.ini` file that can be read by the [configparser](https://docs.python.org/3/library/configparser.html) standard library. You may change the values in this file to adjust it to your setup, but **DO NOT CHANGE THE NAMES**, otherwise the functions will not be able to read the values. The example in [Quick Start](#quick-start) could now be rewritten as

```py
# Import everything from slmcontrol
from slmcontrol import *

# Initialize our slm display
slm = SLMdisplay(monitor=1)

# Path to the configuration file
config_path = 'config.ini'

# The beam that we wish to produce. 
# In this case, a Laguerre-Gaussian with p=0 and l=1
# We don't need the x,y, as it is calculated from the configuration file
desired = lg(config_path, 0, 1, .5)

# Generates the hologram that will be shown in the SLM using only the configuration file
# Notice that we don't even need to define the incoming beam as its wais is already defined in the configuration file
holo = generate_hologram(desired, config_path)

# Displays the generated hologram
slm.updateArray(holo)
```

This is now much cleaner!

!!! Note
    Notice that the `generate_hologram` which is called here is takes only 2 arguments, instead of 9, such as in the section [Quick Start](#quick-start). This is what is called [Multiple Dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), and is a paradigm supported by languages such as [Julia](https://julialang.org/). This is not possible in vanilla Python, but there are packages that enable it. Here, we use [multimethod](https://pypi.org/project/multimethod/).

## Creating a GUI with Jupyter Notebooks

Jupyter Notebooks are a great way to use this package, as the interactive environment makes it easy to quickly change configurations. In this tutorial, we will leverage this tool in order to make a GUI (Graphical User Interface). In order to do this, we will use the [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/) package, that allows us to program UI elements in notebooks. We will also use [matplotlib](https://matplotlib.org/stable/) for some plots.

In the first cell, we will simply import the things that we need and initialize and [SLMdisplay][src.slmcontrol.slm.SLMdisplay] object:
```py
from slmcontrol import *
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import configparser

slm = SLMdisplay(monitor=1)
```

Now, we give the path of our [configuration file](#configuration-file) and construct the grid and the incoming beam:
```py
config_path = 'config.ini'
config = configparser.ConfigParser()
config.read(config_path)
x,y = build_grid(config_path)
incoming = hg(x,y,0,0,config['incoming'].getfloat('waist'))
```

Here, we define the UI elements, such as dropdown menus, that will control some parameters. We also define the function that they will interact with:
```py
xperiod = widgets.BoundedFloatText(
    value=config['grating'].getfloat('xperiod'),
    min=-100.0,
    max=100.0,
    step=0.1,
    description='x period',
    disabled=False
)

yperiod = widgets.BoundedFloatText(
    value=config['grating'].getfloat('yperiod'),
    min=-100.0,
    max=100.0,
    step=0.1,
    description='y period',
    disabled=False
)

xoffset = widgets.BoundedIntText(
    value=config['incoming'].getint('xoffset'),
    min=-10000,
    max=10000,
    description='x offset',
    disabled=False
)

yoffset = widgets.BoundedIntText(
    value=config['incoming'].getint('yoffset'),
    min=-10000,
    max=10000,
    description='y offset',
    disabled=False
)

mode = widgets.Dropdown(
    options=['Laguerre-Gauss', 'Hermite-Gauss', 'Diagonal Hermite-Gauss'],
    value='Laguerre-Gauss',
    description='Mode',
    disabled=False,
)

method = widgets.Dropdown(
    options=['bessel1', 'sinc'],
    value='bessel1',
    description='Algorithm',
    disabled=False,
)

idx1 = widgets.BoundedIntText(
    value=1,
    min=-100,
    max=100,
    description='p (m)',
    disabled=False
)

idx2 = widgets.BoundedIntText(
    value=1,
    min=-100,
    max=100,
    description='l (n)',
    disabled=False
)

waist = widgets.BoundedFloatText(
    value=0.5,
    min=.01,
    max=5.0,
    step=0.01,
    description='Waist',
    disabled=False
)

def f(mode,idx1,idx2,waist,xperiod,yperiod,xoffset,yoffset,method):
    if mode == 'Laguerre-Gauss':
        desired = lg(config_path, idx1,idx2, waist)
    elif mode == 'Hermite-Gauss':
        desired = hg(config_path, idx1,idx2, waist)
    elif mode == 'Diagonal Hermite-Gauss':
        desired = diagonal_hg(config_path, idx1,idx2, waist)
    else:
        raise ValueError("Invalid mode.")
    holo = generate_hologram(desired,incoming,x,y,config['slm'].getint('max_modulation'),xperiod,yperiod,xoffset,yoffset,method=method)
    slm.updateArray(holo)
    plt.imshow(np.abs(desired)**2,cmap='jet')
```

Finally, we call the `interact` function from `ipywidgets`:
```py
interact(f,mode=mode,idx1=idx1,idx2=idx2,waist=waist,xperiod=xperiod,yperiod=yperiod,xoffset=xoffset,yoffset=yoffset,method=method);
```