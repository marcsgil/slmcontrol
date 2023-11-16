# Index

`slmcontrol` is a package that allows one to control a Spatial Light Modulator (SLM) with a simple an intuitive syntax.

If you use this package in your research, please cite it and give it a star on GitHub.

## Links

- **PyPI:** [https://pypi.org/project/slmcontrol/]()
- **Documentation:** [https://marcsgil.github.io/slmcontrol/]()
- **Source code:** [https://github.com/marcsgil/slmcontrol/tree/main]()

## Instalation

To install this package, run

```
pip install slmcontrol
```

on a terminal.

**Important**: If you are running Linux, you need to manually install [wxPython](https://wxpython.org/pages/downloads/).

## Table Of Contents

1. In [Tutorials](tutorials.md), you can find a Quick-Start guide, how to setup a configuration file, and how to make a GUI (Graphical User Interface) using Jupyter Notebooks.

2. In [Reference](reference.md) you can find a detailed description of all the exported functions.

3. In [Explanation](explanation.md) you can find a discussion of the working principles of a SLM, and how to build an experimental setup.

## Authors

Developed by PhD students at the Quantum Optics Laboratory of Universidade Federal Fluminense:

- Marcos Gil
- André Junior
- Altilano Barbosa
- Braian Pinheiro

The part of the code that sends image to the SLM (`slm.py`) was taken from the repository [pyslm](https://github.com/wavefrontshaping/slmPy) from Sébastien M. Popoff, with only minor modifications.