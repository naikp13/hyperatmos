import numpy as np
import pytest

from hyperatmos import Geometry, RadiativeTransferModel, SpectralDatabase, preset, standard_atmosphere


@pytest.fixture(scope="session")
def db():
    return SpectralDatabase.build(standard_atmosphere(), (400.0, 2500.0), dnu=0.1)


@pytest.fixture(scope="session")
def rtm(db):
    return RadiativeTransferModel(db, preset("EMIT"), Geometry(35.0, 5.0, 30.0))


@pytest.fixture
def rng():
    return np.random.default_rng(42)
