import pytest

from kmu_rating import CalibrationConfig, KMURatingModel, ModelConfig, generate_kmu_data


@pytest.fixture(scope="session")
def data():
    return generate_kmu_data(15000, seed=7)


@pytest.fixture(scope="session")
def dev(data):
    return data[data.beobachtungsjahr <= 2022].copy()


@pytest.fixture(scope="session")
def oot(data):
    return data[data.beobachtungsjahr == 2023].copy()


@pytest.fixture(scope="session")
def model(dev):
    cfg = ModelConfig(calibration=CalibrationConfig(central_tendency=0.03, moc=0.0))
    return KMURatingModel(cfg).fit(dev)
