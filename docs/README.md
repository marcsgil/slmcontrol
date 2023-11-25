# slmcontrol

This is a package that allows one to control a Spatial Light Modulator (SLM) with a simple an intuitive syntax.

If you use this package in your research, please cite it and give it a star on GitHub.

## Links

- [**PyPI**](https://pypi.org/project/slmcontrol/)
- [**Documentation**](https://marcsgil.github.io/slmcontrol/)
- [**Source code**](https://github.com/marcsgil/slmcontrol/tree/main)

## Instalation

To install this package, run

```
pip install slmcontrol
```

on a terminal.

It will work of the box for Window. That is also the expected behavior for MacOS, but it has not been tested. For Linux, you will need to manually install [wxPython](https://wxpython.org/pages/downloads/) before using this package.

If you encounter the error `libSDL2-2.0.so.0: cannot open shared object file: No such file or directory`, you need to install the package `libsdl2-2.0-0` (on Ubuntu, run `sudo apt install libsdl2-dev`).

If you encounter any problems, please open an issue on GitHub.

## Authors

Developed by PhD students at the Quantum Optics Laboratory of Universidade Federal Fluminense:

- Marcos Gil
- André Junior
- Altilano Barbosa
- Braian Pinheiro

The part of the code that sends image to the SLM (`slm.py`) was taken from the repository [pyslm](https://github.com/wavefrontshaping/slmPy) from Sébastien M. Popoff, with only minor modifications.