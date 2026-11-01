"""Physical constants and unit conversions used throughout hyperatmos.

Gas columns are handled in molecules/cm^2 internally. Water vapour is
reported in g/cm^2 (precipitable centimetres, as in ISOFIT's ``H2OSTR``),
ozone in Dobson units, and other gases as scale factors of the background
column.
"""

from __future__ import annotations

K_BOLTZMANN = 1.380649e-23  # J/K
N_AVOGADRO = 6.02214076e23  # 1/mol
C2 = 1.4387769  # second radiation constant hc/k [cm K]

MOLAR_MASS = {  # kg/mol
    "CH4": 16.04e-3,
    "CO2": 44.01e-3,
    "H2O": 18.015e-3,
    "NO2": 46.0055e-3,
    "SO2": 64.066e-3,
    "O2": 31.998e-3,
    "O3": 47.997e-3,
}

#: HITRAN molecule ids, used when parsing .par line lists.
HITRAN_ID = {"H2O": 1, "CO2": 2, "O3": 3, "CH4": 6, "O2": 7, "SO2": 9, "NO2": 10}


def air_number_density(p_pa: float = 101325.0, t_k: float = 288.15) -> float:
    """Air number density [molecules/cm^3]."""
    return p_pa / (K_BOLTZMANN * t_k) * 1e-6


DOBSON_UNIT = 2.6867e16  # molecules/cm^2


def h2o_g_cm2_to_molec_cm2(w):
    """Precipitable water [g/cm^2] to molecules/cm^2."""
    return w / (MOLAR_MASS["H2O"] * 1e3) * N_AVOGADRO


def molec_cm2_to_h2o_g_cm2(col):
    """Molecules/cm^2 of H2O to precipitable water [g/cm^2]."""
    return col / N_AVOGADRO * MOLAR_MASS["H2O"] * 1e3


def nm_to_wavenumber(wl_nm):
    """Wavelength [nm] to wavenumber [cm^-1]."""
    return 1e7 / wl_nm


def wavenumber_to_nm(nu):
    """Wavenumber [cm^-1] to wavelength [nm]."""
    return 1e7 / nu
