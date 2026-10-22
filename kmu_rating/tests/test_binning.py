import numpy as np
import pandas as pd
import pytest

from kmu_rating.binning import MISSING_LABEL, FeatureBinning
from kmu_rating.config import BinningConfig


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    y = rng.binomial(1, 1 / (1 + np.exp(3 + 1.2 * x)))
    return pd.Series(x), y


def test_monotone_decreasing_badrate(xy):
    x, y = xy
    b = FeatureBinning("x", "numeric", BinningConfig()).fit(x, y)
    rates = b.table["ausfallrate"].to_numpy()
    assert np.all(np.diff(rates) <= 0)
    assert np.all(np.diff(b.woe) >= 0)          # höheres x -> besser -> höheres WoE
    assert len(b.woe) <= 6 and b.iv > 0.3
    assert b.table["anteil"].min() >= 0.05 - 1e-9


def test_iv_formula(xy):
    x, y = xy
    b = FeatureBinning("x", "numeric", BinningConfig(woe_smoothing=0.0)).fit(x, y)
    t = b.table
    pg = (t.anzahl - t.ausfaelle) / (t.anzahl - t.ausfaelle).sum()
    pb = t.ausfaelle / t.ausfaelle.sum()
    assert b.iv == pytest.approx(float(((pg - pb) * np.log(pg / pb)).sum()))


def test_transform_bin_boundaries(xy):
    x, y = xy
    b = FeatureBinning("x", "numeric", BinningConfig()).fit(x, y)
    c = b.cuts[0]
    w = b.transform(pd.Series([c, np.nextafter(c, np.inf)]))
    assert w[0] == b.woe[0] and w[1] == b.woe[1]   # rechts-geschlossene Intervalle


def test_missing_own_bin_and_conservative(xy):
    x, y = xy
    x2 = x.copy()
    x2.iloc[:3000] = np.nan
    b = FeatureBinning("x", "numeric", BinningConfig()).fit(x2, y)
    assert b.missing_own_bin and MISSING_LABEL in b.table["klasse"].tolist()
    x3 = x.copy()
    x3.iloc[:20] = np.nan
    b3 = FeatureBinning("x", "numeric", BinningConfig()).fit(x3, y)
    assert not b3.missing_own_bin
    assert b3.transform(pd.Series([np.nan]))[0] == pytest.approx(b3.woe.min())


def test_categorical_rare_and_unseen():
    rng = np.random.default_rng(1)
    cats = rng.choice(["A", "B", "C", "selten"], size=10000, p=[.45, .3, .24, .01])
    pbad = pd.Series(cats).map({"A": .01, "B": .04, "C": .08, "selten": .2}).to_numpy()
    y = rng.binomial(1, pbad)
    b = FeatureBinning("k", "categorical", BinningConfig()).fit(pd.Series(cats), y)
    w = b.transform(pd.Series(["A", "C", "NEU", None]))
    assert w[0] > w[1]
    assert np.isfinite(w[2]) and np.isfinite(w[3])
    assert len(b.woe) <= 6


def test_non_monotonic_override():
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, 30000)
    y = rng.binomial(1, 0.01 + 0.15 * x ** 2)
    cfg = BinningConfig(monotonic_overrides={"u": False})
    mono = FeatureBinning("u", "numeric", BinningConfig()).fit(pd.Series(x), y)
    free = FeatureBinning("u", "numeric", cfg, monotonic=False).fit(pd.Series(x), y)
    assert free.iv > mono.iv
