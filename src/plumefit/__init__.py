"""plumefit: trace-gas plume retrievals for imaging spectrometers, built to sit next to ISOFIT."""

from .atmosphere import Atmosphere, standard_atmosphere
from .forward import ForwardModel, Geometry
from .instrument import Instrument, preset, uniform_instrument
from .spectroscopy import SpectralDatabase

__version__ = "0.1.0"

__all__ = [
    "Atmosphere",
    "ForwardModel",
    "Geometry",
    "Instrument",
    "SpectralDatabase",
    "preset",
    "standard_atmosphere",
    "uniform_instrument",
]
