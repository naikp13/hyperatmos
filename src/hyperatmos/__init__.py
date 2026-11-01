"""hyperatmos: radiative transfer and atmospheric correction for drone and satellite imaging spectrometers."""

from .atmcorr import AtmosphericCorrection
from .atmosphere import Atmosphere, standard_atmosphere
from .forward import AtmosphereSurfaceModel
from .instrument import Instrument, preset, uniform_instrument
from .rtm import Aerosol, Geometry, RadiativeTransferModel
from .spectroscopy import SpectralDatabase

__version__ = "0.1.0"

__all__ = [
    "Aerosol",
    "Atmosphere",
    "AtmosphereSurfaceModel",
    "AtmosphericCorrection",
    "Geometry",
    "Instrument",
    "RadiativeTransferModel",
    "SpectralDatabase",
    "preset",
    "standard_atmosphere",
    "uniform_instrument",
]
