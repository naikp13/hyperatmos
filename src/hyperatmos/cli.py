"""Command-line interface: ``hyperatmos <command> --help``."""

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


def _rtm(args, hdr=None):
    from .rtm import Aerosol, Geometry, RadiativeTransferModel
    from .spectroscopy import SpectralDatabase

    db = SpectralDatabase.load(args.db)
    geom = Geometry(args.sza, args.vza, args.raa, args.altitude)
    return RadiativeTransferModel(db, _instrument_from_args(args, hdr), geom, Aerosol(angstrom=args.angstrom))


def _add_common(p):
    p.add_argument("--db", required=True, help="spectral database (.npz) from 'build-db'")
    p.add_argument("--sza", type=float, default=30.0, help="solar zenith angle [deg]")
    p.add_argument("--vza", type=float, default=0.0, help="view zenith angle [deg]")
    p.add_argument("--raa", type=float, default=0.0, help="relative azimuth [deg], 0 = backscatter side")
    p.add_argument("--altitude", type=float, default=None, help="sensor altitude above ground [m]; omit for satellites")
    p.add_argument("--sensor", default="EMIT", help="instrument preset if no wavelength info")
    p.add_argument("--wavelength-file", help="ISOFIT wavelength file")
    p.add_argument("--snr", type=float, default=300.0, help="SNR at reference radiance")
    p.add_argument("--angstrom", type=float, default=1.3, help="aerosol Angstrom exponent")


def cmd_build_db(args):
    from .atmosphere import standard_atmosphere
    from .spectroscopy import SpectralDatabase, read_hitran_par

    atm = standard_atmosphere(h2o_g_cm2=args.h2o, o3_du=args.o3_du, co2_ppm=args.co2_ppm, ch4_ppm=args.ch4_ppm)
    linelists = {}
    for item in args.hitran or []:
        gas, path = item.split("=", 1)
        linelists[gas] = read_hitran_par(path, gas)
    db = SpectralDatabase.build(atm, tuple(args.wl), tuple(args.gases), linelists, dnu=args.dnu)
    db.save(args.out)
    print(f"wrote {args.out} ({db.nu.size} spectral points, synthetic={db.synthetic})")


def cmd_simulate(args):
    from .io import write_envi
    from .simulate import simulate_scene

    rtm = _rtm(args)
    sc = simulate_scene(rtm, (args.rows, args.cols), h2o_mean=args.h2o, aod_mean=args.aod, seed=args.seed)
    fw = rtm.instrument.fwhm_nm
    write_envi(
        f"{args.out}_rdn", sc.radiance, rtm.wavelength_nm, fwhm_nm=fw, description="hyperatmos synthetic radiance"
    )
    write_envi(f"{args.out}_true_rfl", sc.reflectance, rtm.wavelength_nm, fwhm_nm=fw)
    write_envi(f"{args.out}_true_h2o", sc.h2o_g_cm2, band_names=["H2O g cm-2"])
    print(json.dumps({"radiance": f"{args.out}_rdn", "h2o_mean": float(sc.h2o_g_cm2.mean()), "aod550": args.aod}))


def cmd_correct(args):
    from .atmcorr import AtmosphericCorrection
    from .io import read_envi
    from .isofit_interop import write_isofit_outputs

    rdn, hdr = read_envi(args.radiance)
    rtm = _rtm(args, hdr)
    res = AtmosphericCorrection(rtm).correct(rdn, aod550=args.aod)
    rfl, state = write_isofit_outputs(
        args.out, res.reflectance, res.wavelength_nm, res.h2o_g_cm2, res.aod550, rtm.instrument.fwhm_nm
    )
    print(
        json.dumps(
            {"rfl": str(rfl), "state": str(state), "aod550": res.aod550, "h2o_median": float(np.median(res.h2o_g_cm2))}
        )
    )


def cmd_design(args):
    from .design import trade_study
    from .rtm import Geometry
    from .spectroscopy import SpectralDatabase

    db = SpectralDatabase.load(args.db)
    gases = tuple(args.gases or ())
    grid = trade_study(
        db,
        Geometry(args.sza, args.vza, args.raa, args.altitude),
        np.array(args.fwhm_grid),
        np.array(args.snr_grid),
        quantity=args.quantity,
        wl_range=tuple(args.wl),
        reflectance=args.reflectance,
        gases=gases,
    )
    print(f"posterior 1-sigma of {args.quantity}")
    print("FWHM\\SNR " + " ".join(f"{s:>8.0f}" for s in args.snr_grid))
    for fw, row in zip(args.fwhm_grid, grid, strict=True):
        print(f"{fw:8.1f} " + " ".join(f"{v:8.4f}" for v in row))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hyperatmos", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-db", help="compute the high-resolution gas optical-depth database")
    p.add_argument("--out", default="hyperatmos_db.npz")
    p.add_argument("--wl", type=float, nargs=2, default=[380.0, 2500.0], metavar=("MIN", "MAX"))
    p.add_argument("--gases", nargs="+", default=["H2O", "O2", "CO2", "CH4", "O3"])
    p.add_argument(
        "--hitran",
        nargs="*",
        metavar="GAS=FILE.par",
        help="HITRAN line lists; gases without one use synthetic demo spectroscopy",
    )
    p.add_argument("--dnu", type=float, default=0.05, help="spectral grid step [cm^-1]")
    p.add_argument("--h2o", type=float, default=1.5, help="reference water vapour [g/cm^2]")
    p.add_argument("--o3-du", type=float, default=300.0)
    p.add_argument("--co2-ppm", type=float, default=420.0)
    p.add_argument("--ch4-ppm", type=float, default=1.92)
    p.set_defaults(func=cmd_build_db)

    p = sub.add_parser("simulate", help="simulate a radiance scene with known surface and atmosphere")
    _add_common(p)
    p.add_argument("--out", default="scene")
    p.add_argument("--rows", type=int, default=60)
    p.add_argument("--cols", type=int, default=60)
    p.add_argument("--h2o", type=float, default=1.5, help="mean water vapour [g/cm^2]")
    p.add_argument("--aod", type=float, default=0.12, help="AOD at 550 nm")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("correct", help="atmospheric correction of an ENVI radiance cube (ISOFIT-style outputs)")
    _add_common(p)
    p.add_argument("radiance")
    p.add_argument("--out", required=True, help="output prefix; writes <out>_rfl and <out>_state")
    p.add_argument("--aod", type=float, default=None, help="AOD at 550 nm; estimated from dark vegetation if omitted")
    p.set_defaults(func=cmd_correct)

    p = sub.add_parser("design", help="instrument trade study (FWHM x SNR)")
    _add_common(p)
    p.add_argument("--quantity", default="H2O", help="H2O, AOD550 or scale_<GAS>")
    p.add_argument("--gases", nargs="*", help="extra gases to retrieve as column scales, e.g. CH4")
    p.add_argument("--fwhm-grid", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    p.add_argument("--snr-grid", type=float, nargs="+", default=[100.0, 300.0, 600.0])
    p.add_argument("--wl", type=float, nargs=2, default=[850.0, 1250.0])
    p.add_argument("--reflectance", type=float, default=0.25)
    p.set_defaults(func=cmd_design)

    args = parser.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
