# Explanation

## Background and Motivation

Spatial Light Modulators (SLMs) represent a powerful technology for precise light manipulation in optical systems. The `slmcontrol` package was developed to address the complexity of working with these devices, providing a streamlined Python interface.

## Core Capabilities

The package is designed around three primary functional components:

1. **Display Capabilities**: The [`SLMDisplay`][src.slmcontrol.slm.SLMDisplay] class provides a high-level interface for displaying images on the SLM. 
The main method `updateArray` allows users to send a 2D array of pixel values to the SLM, which is then rendered on the device. The image is represented as a 2D array of Unsigned 8-bit integers (uint8), where each value corresponds to a phase shift imposed by the corresponding pixel. The image can in principle be anything, but, in order to produce a structured light mode, one will usually use a hologram that encodes the desired phase pattern, as described in the next section.


Direct control of SLM displays, enabling precise pixel-level phase modulation. This allows for accurate wavefront shaping and complex light field generation.

2. **Hologram Generation**: Algorithmic methods for creating phase patterns that transform incoming light into desired spatial configurations. The approach supports various beam types, including complex modes like Laguerre-Gaussian and Hermite-Gaussian beams.

3. **Structural Flexibility**: While pre-defined structures are provided for convenience, the package is fundamentally designed to be extensible. Researchers can easily implement custom beam structures and phase manipulation techniques.