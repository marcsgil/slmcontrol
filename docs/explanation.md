# Explanation

## Background and Motivation

Spatial Light Modulators (SLMs) represent a powerful technology for precise light manipulation in optical systems. The `slmcontrol` package was developed to address the complexity of working with these devices, providing a streamlined Python interface.

## Core Capabilities

The package is designed around three primary functional components:

1. **Display Capabilities**: The [`SLMDisplay`][src.slmcontrol.slm.SLMDisplay] class provides a high-level interface for displaying images on the SLM. 
The main method [`updateArray`][src.slmcontrol.slm.SLMDisplay.updateArray] allows users to send a 2D array of pixel values to the SLM, which is then rendered on the device. The image is represented as a 2D array of Unsigned 8-bit integers (uint8), where each value corresponds to a phase shift imposed by the corresponding pixel. The image can in principle be anything, but, in order to produce a structured light mode, one will usually use a hologram that encodes the desired phase pattern, as described in the next section.

2. **Hologram Generation**: The [`generate_hologram`][src.slmcontrol.hologram.generate_hologram] function calculates the phase patterns needed to transform an incoming beam (typically a plane wave or a Gaussian) into a desired output beam. The output can be any complex field, and is not restricted to the structured light modes provided in the package. 

3. **Pre-defined structured modes**: We provide a set of pre-defined functions for generating common optical fields, such as Laguerre-Gaussian and Hermite-Gaussian modes, as well as various apertures and lenses. These fields can be used in the `desired` argument of the [`generate_hologram`][src.slmcontrol.hologram.generate_hologram] function. Nonetheless, the user is free to define their own desired field, which can be any complex field. The pre-defined functions are designed to be user-friendly, allowing researchers to quickly generate the desired beam profiles without delving into the underlying mathematics. Here is a list of the pre-defined functions available in the package:
    1. **Gaussian Beam Variations**:

        - [`lg`][src.slmcontrol.structures.lg]: Laguerre-Gaussian modes characterized by radial (p) and azimuthal (l) indices.
        - [`hg`][src.slmcontrol.structures.hg]: Hermite-Gaussian modes characterized by vertical (m) and horizontal (n) indices.
        - [`diagonal_hg`][src.slmcontrol.structures.diagonal_hg]: Diagonal Hermite-Gaussian modes.

    2. **Optical Elements**:

        - [`lens`][src.slmcontrol.structures.lens]: Phase function for a cylindrical lens with separate focal lengths in x and y directions.
        - [`tilted_lens`][src.slmcontrol.structures.tilted_lens]: Phase function for a tilted spherical lens.

    3. **Apertures**:

        - [`rectangular_apperture`][src.slmcontrol.structures.rectangular_apperture]: Rectangular aperture with specified dimensions.
        - [`square`][src.slmcontrol.structures.square]: Square aperture with a given side length.
        - [`single_slit`][src.slmcontrol.structures.single_slit] and [`double_slit`][src.slmcontrol.structures.double_slit]: Vertical slit patterns.
        - [`pupil`][src.slmcontrol.structures.pupil]: Circular pupil with specified radius.
        - [`triangle`][src.slmcontrol.structures.triangle]: Equilateral triangular aperture.

    4. **Wavefront Correction**:

        - [`zernike`][src.slmcontrol.zernike.zernike]: Zernike polynomials for wavefront aberration correction, characterized by radial (n) and azimuthal (m) orders.

### Implementation Architecture

The `slmcontrol` package is built as a Python wrapper around Julia libraries, leveraging the high-performance numerical capabilities of Julia:

- The display functionality uses the [`SpatialLightModulator`](https://github.com/marcsgil/SpatialLightModulator.jl) Julia package. This, in turn, relies on a wrapper of [OpenGL](https://www.opengl.org/) to display the images on the SLM. The package is designed to be agnostic to the specific SLM hardware, allowing for flexibility in device choice.
- The structured light generation and hologram calculation use the [`StructuredLight`](https://github.com/marcsgil/StructuredLight.jl) Julia package. This allows for simple and efficient generation of structured light modes, working both on CPU, with multithreading, and on GPUs.
- The Python-Julia bridge is implemented using the [`juliacall`](https://github.com/JuliaPy/PythonCall.jl) package.

## Applications

The `slmcontrol` package can be used in a variety of applications:

1. **Optical Tweezers**: Generate specialized beam profiles for trapping and manipulating microscopic particles.
2. **Quantum Optics**: Create structured light fields with orbital angular momentum for quantum information experiments.
3. **Microscopy**: Implement wavefront correction and beam shaping for advanced microscopy techniques.
4. **Education**: Demonstrate principles of optics and wave propagation in a hands-on manner.