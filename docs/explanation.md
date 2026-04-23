# Explanation

## Background and Motivation

Spatial Light Modulators (SLMs) represent a powerful technology for precise light manipulation in optical systems. The `slmcontrol` package was developed to address the complexity of working with these devices, providing a streamlined Python interface.

## Core Capabilities

The package is designed around three primary functional components:

1. **Display Capabilities**: The [`SLMDisplay`][src.slmcontrol.slm.SLMDisplay] class provides a high-level interface for displaying images on the SLM. 
The main method [`updateArray`][src.slmcontrol.slm.SLMDisplay.updateArray] allows users to send a 2D array of pixel values to the SLM, which is then rendered on the device. The image is represented as a 2D array of Unsigned 8-bit integers (uint8), where each value corresponds to a phase shift imposed by the corresponding pixel. The image can in principle be anything, but, in order to produce a structured light mode, one will usually use a hologram that encodes the desired phase pattern, as described in the next section.

2. **Hologram Generation**: The [`generate_hologram`][src.slmcontrol.hologram.generate_hologram] function calculates the phase patterns needed to transform an incoming beam (typically a plane wave or a Gaussian) into a desired output beam. The output can be any complex field, and is not restricted to the structured light modes provided in the package. 

3. **Pre-defined structured modes**: We provide a set of pre-defined functions for generating common optical fields, such as Laguerre-Gaussian and Hermite-Gaussian modes, as well as various apertures and lenses. These fields can be used in the [`generate_hologram`][src.slmcontrol.hologram.generate_hologram] function. Nonetheless, the user is free to define their own desired field, which can be any complex field. The pre-defined functions are designed to be user-friendly, allowing researchers to quickly generate the desired beam profiles without delving into the underlying mathematics. Here is a list of the pre-defined functions available in the package:
    1. **Gaussian Beam Variations**:

        - [`lg`][src.slmcontrol.structures.lg]: Laguerre-Gaussian modes characterized by radial (p) and azimuthal (l) indices.
        - [`hg`][src.slmcontrol.structures.hg]: Hermite-Gaussian modes characterized by vertical (m) and horizontal (n) indices.
        - [`diagonal_hg`][src.slmcontrol.structures.diagonal_hg]: Diagonal Hermite-Gaussian modes.

    2. **Optical Elements**:

        - [`lens`][src.slmcontrol.structures.lens]: Phase function for a spherical lens with separate focal lengths in x and y directions.

    3. **Apertures**:

        - [`rectangular_aperture`][src.slmcontrol.masks.rectangular_aperture]: Rectangular aperture with specified dimensions.
        - [`square`][src.slmcontrol.masks.square]: Square aperture with a given side length.
        - [`single_slit`][src.slmcontrol.masks.single_slit] and [`double_slit`][src.slmcontrol.masks.double_slit]: Vertical slit patterns.
        - [`pupil`][src.slmcontrol.masks.pupil]: Circular pupil with specified radius.
        - [`triangle`][src.slmcontrol.masks.triangle]: Equilateral triangular aperture.


### Implementation Architecture

The `slmcontrol` package is is structured into several modules, each responsible for a specific aspect of SLM control:

1. **SLM Module** (`src.slmcontrol.slm`): Contains the `SLMDisplay` class, which manages the connection to the SLM hardware and handles image display. This is implemented using OpenCV for cross-platform compatibility, ensuring that the package can work on various operating systems. For setups where the script runs on a different machine than the SLM (e.g. via SSH), a client-server mode is available — see the [Remote Control](remote.md) guide.

2. **Hologram Module** (`src.slmcontrol.hologram`): Implements the `generate_hologram` function, which calculates the phase patterns required to produce the desired output beam from a given input beam. 

3. **Structures Module** (`src.slmcontrol.structures`): Provides functions for generating common structured light modes, such as Laguerre-Gaussian and Hermite-Gaussian beams.

4. **Masks Module** (`src.slmcontrol.masks`): Contains functions for creating various aperture shapes and optical elements that can be used in conjunction with the hologram generation.

## Applications

The `slmcontrol` package can be used in a variety of applications:

1. **Optical Tweezers**: Generate specialized beam profiles for trapping and manipulating microscopic particles.
2. **Quantum Optics**: Create structured light fields with orbital angular momentum for quantum information experiments.
3. **Microscopy**: Implement wavefront correction and beam shaping for advanced microscopy techniques.
4. **Education**: Demonstrate principles of optics and wave propagation in a hands-on manner.