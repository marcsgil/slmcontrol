from slmcontrol.slm import SLMDisplay
from slmcontrol.server import SLMServer
from slmcontrol.hologram import generate_hologram
from slmcontrol.structures import lg, hg, diagonal_hg
from slmcontrol.masks import (
    rectangular_aperture,
    square,
    single_slit,
    double_slit,
    pupil,
    triangle,
)

__all__ = [
    "SLMDisplay",
    "SLMServer",
    "generate_hologram",
    "lg",
    "hg",
    "diagonal_hg",
    "rectangular_aperture",
    "square",
    "single_slit",
    "double_slit",
    "pupil",
    "triangle",
]
