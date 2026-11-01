"""Minimal ENVI raster I/O (the format ISOFIT reads and writes).

Kept dependency-free so hyperatmos can read ISOFIT products without the
``spectral`` package. Arrays are returned as (lines, samples, bands).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}
_CODES = {np.dtype(v): k for k, v in _DTYPES.items()}


def _header_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix == ".hdr":
        return p
    for cand in (Path(str(p) + ".hdr"), p.with_suffix(".hdr")):
        if cand.exists():
            return cand
    return Path(str(p) + ".hdr")


def _data_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix != ".hdr":
        return p
    for ext in ("", ".img", ".dat", ".bin"):
        cand = p.with_suffix(ext)
        if cand.exists():
            return cand
    return p.with_suffix("")


def read_envi_header(path: str | Path) -> dict:
    text = _header_path(path).read_text()
    if not text.lstrip().startswith("ENVI"):
        raise ValueError(f"{path} is not an ENVI header")
    hdr = {}
    for m in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text, flags=re.MULTILINE):
        key, val = m.group(1).strip().lower(), m.group(2).strip()
        if val.startswith("{"):
            items = [v.strip() for v in val[1:-1].split(",")]
            try:
                hdr[key] = [float(v) for v in items if v]
            except ValueError:
                hdr[key] = items
        else:
            hdr[key] = val
    return hdr


def read_envi(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read an ENVI image into memory as (lines, samples, bands)."""
    hdr = read_envi_header(path)
    lines, samples, bands = int(hdr["lines"]), int(hdr["samples"]), int(hdr["bands"])
    dtype = np.dtype(_DTYPES[int(hdr["data type"])])
    if int(hdr.get("byte order", 0)) == 1:
        dtype = dtype.newbyteorder(">")
    offset = int(hdr.get("header offset", 0))
    raw = np.fromfile(_data_path(path), dtype=dtype, offset=offset)
    il = hdr.get("interleave", "bil").lower()
    if il == "bip":
        arr = raw.reshape(lines, samples, bands)
    elif il == "bil":
        arr = raw.reshape(lines, bands, samples).transpose(0, 2, 1)
    elif il == "bsq":
        arr = raw.reshape(bands, lines, samples).transpose(1, 2, 0)
    else:
        raise ValueError(f"Unknown interleave {il}")
    return np.ascontiguousarray(arr).astype(np.float64 if dtype.kind == "f" else dtype.newbyteorder("=")), hdr


def write_envi(
    path: str | Path,
    data: np.ndarray,
    wavelength_nm: np.ndarray | None = None,
    band_names: list[str] | None = None,
    fwhm_nm: np.ndarray | None = None,
    interleave: str = "bil",
    description: str = "hyperatmos output",
) -> None:
    """Write a (lines, samples, bands) or (lines, samples) array as float32 ENVI."""
    p = Path(path)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    lines, samples, bands = arr.shape
    order = {"bip": (0, 1, 2), "bil": (0, 2, 1), "bsq": (2, 0, 1)}[interleave]
    arr.transpose(order).tofile(p)
    hdr = [
        "ENVI",
        f"description = {{{description}}}",
        f"samples = {samples}",
        f"lines = {lines}",
        f"bands = {bands}",
        "header offset = 0",
        "file type = ENVI Standard",
        f"data type = {_CODES[np.dtype(np.float32)]}",
        f"interleave = {interleave}",
        "byte order = 0",
        "data ignore value = -9999",
    ]
    if wavelength_nm is not None:
        hdr.append("wavelength units = Nanometers")
        hdr.append("wavelength = {" + ", ".join(f"{w:.4f}" for w in wavelength_nm) + "}")
    if fwhm_nm is not None:
        hdr.append("fwhm = {" + ", ".join(f"{w:.4f}" for w in fwhm_nm) + "}")
    if band_names is not None:
        hdr.append("band names = {" + ", ".join(band_names) + "}")
    Path(str(p) + ".hdr").write_text("\n".join(hdr) + "\n")


def header_wavelengths_nm(hdr: dict) -> np.ndarray | None:
    """Band centres from an ENVI header, converted to nm when given in microns."""
    if "wavelength" not in hdr:
        return None
    wl = np.asarray(hdr["wavelength"], dtype=float)
    units = str(hdr.get("wavelength units", "")).lower()
    if units.startswith("micro") or wl.max() < 100:
        wl = wl * 1e3
    return wl
