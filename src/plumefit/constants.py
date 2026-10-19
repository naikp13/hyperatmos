"""Physical constants and unit conversions used throughout plumefit.

Internally, gas column enhancements are carried in **ppm·m** (the convention
used by AVIRIS-NG / EMIT methane products) and converted to molecules/cm^2
only where spectroscopy is applied.
"""

from __future__ import annotations

K_BOLTZMANN = 1.380649e-23  # J/K
N_AVOGADRO = 6.02214076e23  # 1/mol
C2 = 1.4387769  # second radiation constant hc/k [cm K]

#: Reference state that defines "ppm·m" (1 ppm of gas over a 1 m path).
#: plumefit uses 1 atm and 288.15 K; change via ``set_ppmm_reference`` if your
#: product convention differs.
P_REF = 101325.0  # Pa
T_REF = 288.15  # K

MOLAR_MASS = {  # kg/mol
    "CH4": 16.04e-3,
    "CO2": 44.01e-3,
    "H2O": 18.015e-3,
    "NO2": 46.0055e-3,
    "SO2": 64.066e-3,
}

#: HITRAN molecule ids, used when parsing .par line lists.
HITRAN_ID = {"H2O": 1, "CO2": 2, "CH4": 6, "NO2": 10, "SO2": 9}


def air_number_density(p_pa: float = P_REF, t_k: float = T_REF) -> float:
    """Air number density [molecules/cm^3]."""
    return p_pa / (K_BOLTZMANN * t_k) * 1e-6


def ppmm_to_molec_cm2(ppmm, p_pa: float = P_REF, t_k: float = T_REF):
    """Convert a column enhancement in ppm·m to molecules/cm^2."""
    return ppmm * 1e-6 * air_number_density(p_pa, t_k) * 100.0


def molec_cm2_to_ppmm(col, p_pa: float = P_REF, t_k: float = T_REF):
    """Convert molecules/cm^2 to ppm·m."""
    return col / (1e-6 * air_number_density(p_pa, t_k) * 100.0)


def ppmm_to_kg_m2(ppmm, gas: str, p_pa: float = P_REF, t_k: float = T_REF):
    """Convert ppm·m to an areal mass density [kg/m^2]."""
    molec_m2 = ppmm_to_molec_cm2(ppmm, p_pa, t_k) * 1e4
    return molec_m2 / N_AVOGADRO * MOLAR_MASS[gas]


def kg_m2_to_ppmm(kg_m2, gas: str, p_pa: float = P_REF, t_k: float = T_REF):
    """Convert an areal mass density [kg/m^2] to ppm·m."""
    molec_cm2 = kg_m2 / MOLAR_MASS[gas] * N_AVOGADRO * 1e-4
    return molec_cm2_to_ppmm(molec_cm2, p_pa, t_k)


def nm_to_wavenumber(wl_nm):
    """Wavelength [nm] to wavenumber [cm^-1]."""
    return 1e7 / wl_nm


def wavenumber_to_nm(nu):
    """Wavenumber [cm^-1] to wavelength [nm]."""
    return 1e7 / nu
