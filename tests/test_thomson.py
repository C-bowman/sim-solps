from numpy import linspace, exp
from sims.instruments.thomson import ThomsonScattering
import pytest


@pytest.fixture
def spatial_data():
    mu = linspace(1.0, 1.4, 21)
    dR = linspace(-0.03, 0.03, 25)

    R = mu[:, None] - dR[None, :]
    sigma = 0.01
    z = (R - mu[:, None]) / sigma
    weights = exp(-0.5 * z**2)
    weights /= weights.sum(axis=1)[:, None]
    return R, z, weights


@pytest.fixture
def measurements():
    R = linspace(1.0, 1.4, 21)
    te = 100 - (R - 1.0) * 200
    ne = 1e19 - (R - 1.0) * 1e19
    return {"te_data": te, "te_err": te * 0.1, "ne_data": ne, "ne_err": ne * 0.1}


def test_measurement_parsing(spatial_data, measurements):
    R, z, weights = spatial_data
    TS = ThomsonScattering(R=R, z=z, weights=weights, measurements=measurements)
