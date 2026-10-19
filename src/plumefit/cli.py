"""Command-line interface: ``plumefit <command> --help``."""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np


def _instrument_from_args(args, hdr=None):
    from .instrument import Instrument, preset
    from .io import header_wavelengths_nm

    if args.wavelength_file:
        return Instrument.from_isofit_wavelength_file(args.wavelength_file, snr_ref=args.snr)
    if hdr is not None and "wavelength" in hdr:
        wl = header_wavelengths_nm(hdr)
        fwhm = np.asarray(hdr["fwhm"], float) if "fwhm" in hdr else np.gradient(wl)
        if fwhm.max() < 1:
            fwhm = fwhm * 1e3
        return Instrument(wl, fwhm, snr_ref=args.snr)
    return preset(args.sensor)


def _geometry(args):
    from .forward import Geometry

    return Geometry(args.sza, args.vza, args.altitude)


def _add_common(p):
    p.add_argument("--db", required=True, help="spectral database (.npz) from 'build-db'")
    p.add_argument("--sza", type=float, default=30.0, help="solar zenith angle [deg]")
    p.add_argument("--vza", type=float, default=0.0, help="view zenith angle [deg]")
    p.add_argument("--altitude", type=float, default=None, help="sensor altitude above ground [m]; omit for satellites")
    p.add_argument("--sensor", default="EMIT", help="instrument preset if no wavelength info")
    p.add_argument("--wavelength-file", help="ISOFIT wavelength file")
    p.add_argument("--snr", type=float, default=300.0, help="SNR at reference radiance")


def cmd_build_db(args):
    from .atmosphere import standard_atmosphere
    from .isofit_bridge import scale_atmosphere_h2o
    from .spectroscopy import SpectralDatabase, read_hitran_par

    atm = standard_atmosphere(ch4_ppm=args.ch4_ppm, co2_ppm=args.co2_ppm)
    if args.h2o_g_cm2:
        scale_atmosphere_h2o(atm, args.h2o_g_cm2)
    linelists = {}
    for item in args.hitran or []:
        gas, path = item.split("=", 1)
        linelists[gas] = read_hitran_par(path, gas)
    db = SpectralDatabase.build(atm, tuple(args.wl), tuple(args.gases), linelists, dnu=args.dnu)
    db.save(args.out)
    print(f"wrote {args.out} ({db.nu.size} spectral points, synthetic={db.synthetic})")


def cmd_simulate(args):
    from .forward import ForwardModel
    from .io import write_envi
    from .simulate import simulate_scene
    from .spectroscopy import SpectralDatabase

    db = SpectralDatabase.load(args.db)
    fm = ForwardModel(db, _instrument_from_args(args), _geometry(args))
    sc = simulate_scene(fm, (args.rows, args.cols), args.pixel_size, args.q, args.wind, args.wind_dir, seed=args.seed)
    write_envi(
        f"{args.out}_rdn",
        sc.radiance,
        fm.wavelength_nm,
        fwhm_nm=fm.instrument.fwhm_nm,
        description="plumefit synthetic radiance",
    )
    write_envi(f"{args.out}_truth", sc.true_enhancement, band_names=["CH4 ppm m"])
    write_envi(f"{args.out}_rfl", sc.reflectance, fm.wavelength_nm)
    print(json.dumps({"radiance": f"{args.out}_rdn", "truth_max_ppmm": float(sc.true_enhancement.max())}))


def cmd_mf(args):
    from .forward import ForwardModel
    from .io import read_envi, write_envi
    from .retrieval import matched_filter
    from .spectroscopy import SpectralDatabase

    rdn, hdr = read_envi(args.radiance)
    db = SpectralDatabase.load(args.db)
    inst = _instrument_from_args(args, hdr)
    fm = ForwardModel(db, inst, _geometry(args), retrieved_gases=(args.gas,))
    k = fm.unit_absorption_spectrum(args.gas)
    res = matched_filter(rdn[..., fm.band_mask], k, column_group=args.column_group)
    write_envi(
        args.out,
        np.stack([res.enhancement, res.uncertainty], -1),
        band_names=[f"{args.gas} ppm m", "uncertainty ppm m"],
    )
    print(f"wrote {args.out}")


def cmd_design(args):
    from .design import trade_study
    from .spectroscopy import SpectralDatabase

    db = SpectralDatabase.load(args.db)
    grid = trade_study(
        db,
        _geometry(args),
        np.array(args.fwhm_grid),
        np.array(args.snr_grid),
        wl_range=tuple(args.wl),
        reflectance=args.reflectance,
    )
    print("1-sigma single-pixel CH4 precision [ppm m]")
    print("FWHM\\SNR " + " ".join(f"{s:>8.0f}" for s in args.snr_grid))
    for fw, row in zip(args.fwhm_grid, grid, strict=True):
        print(f"{fw:8.1f} " + " ".join(f"{v:8.0f}" for v in row))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="plumefit", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-db", help="compute the high-resolution spectral database")
    p.add_argument("--out", default="plumefit_db.npz")
    p.add_argument("--wl", type=float, nargs=2, default=[2100.0, 2450.0], metavar=("MIN", "MAX"))
    p.add_argument("--gases", nargs="+", default=["CH4", "CO2", "H2O"])
    p.add_argument(
        "--hitran",
        nargs="*",
        metavar="GAS=FILE.par",
        help="HITRAN line lists; gases without one use synthetic demo spectroscopy",
    )
    p.add_argument("--dnu", type=float, default=0.02, help="spectral grid step [cm^-1]")
    p.add_argument("--ch4-ppm", type=float, default=1.92)
    p.add_argument("--co2-ppm", type=float, default=420.0)
    p.add_argument("--h2o-g-cm2", type=float, help="total water vapour, e.g. ISOFIT H2OSTR")
    p.set_defaults(func=cmd_build_db)

    p = sub.add_parser("simulate", help="simulate a radiance scene with a CH4 point source")
    _add_common(p)
    p.add_argument("--out", default="scene")
    p.add_argument("--rows", type=int, default=80)
    p.add_argument("--cols", type=int, default=80)
    p.add_argument("--pixel-size", type=float, default=30.0)
    p.add_argument("--q", type=float, default=1000.0, help="emission rate [kg/h]")
    p.add_argument("--wind", type=float, default=4.0, help="wind speed [m/s]")
    p.add_argument("--wind-dir", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("mf", help="matched-filter enhancement map from an ENVI radiance cube")
    _add_common(p)
    p.add_argument("radiance")
    p.add_argument("--out", required=True)
    p.add_argument("--gas", default="CH4")
    p.add_argument("--column-group", type=int, default=1)
    p.set_defaults(func=cmd_mf)

    p = sub.add_parser("design", help="instrument trade study (FWHM x SNR)")
    _add_common(p)
    p.add_argument("--fwhm-grid", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0, 10.0])
    p.add_argument("--snr-grid", type=float, nargs="+", default=[100.0, 200.0, 400.0, 800.0])
    p.add_argument("--wl", type=float, nargs=2, default=[2120.0, 2430.0])
    p.add_argument("--reflectance", type=float, default=0.25)
    p.set_defaults(func=cmd_design)

    args = parser.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
