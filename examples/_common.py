"""Shared setup for the examples: a cached spectral database and output paths."""

from pathlib import Path

from plumefit import SpectralDatabase, standard_atmosphere

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = ROOT / ".cache" / "demo_db_2100_2450.npz"

DEMO_NOTE = "synthetic demo spectroscopy (illustrative only)"


def demo_database() -> SpectralDatabase:
    """Build once, then load from cache. Pass HITRAN line lists to build() for real work."""
    if DB_PATH.exists():
        return SpectralDatabase.load(DB_PATH)
    DB_PATH.parent.mkdir(exist_ok=True)
    db = SpectralDatabase.build(standard_atmosphere(), (2100.0, 2450.0))
    db.save(DB_PATH)
    return db
