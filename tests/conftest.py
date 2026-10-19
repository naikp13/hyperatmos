import numpy as np
import pytest

from plumefit import ForwardModel, Geometry, SpectralDatabase, preset, standard_atmosphere


@pytest.fixture(scope="session")
def db():
    return SpectralDatabase.build(standard_atmosphere(), (2100.0, 2450.0), dnu=0.04)


@pytest.fixture(scope="session")
def fm(db):
    return ForwardModel(db, preset("EMIT"), Geometry(30.0, 0.0))


@pytest.fixture
def rng():
    return np.random.default_rng(42)
