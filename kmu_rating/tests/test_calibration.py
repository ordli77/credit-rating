import numpy as np
import pytest

from kmu_rating.calibration import MasterScale, apply_moc_and_floor, expit, intercept_shift


def test_intercept_shift_hits_target():
    z = np.random.default_rng(0).normal(-4, 1, 5000)
    d = intercept_shift(z, 0.02)
    assert np.mean(expit(z + d)) == pytest.approx(0.02, abs=1e-9)


def test_moc_floor():
    p = apply_moc_and_floor(np.array([0.0001, 0.1, 0.99]), 0.1, 0.0005)
    assert p.tolist() == pytest.approx([0.0005, 0.11, 1.0])


def test_master_scale_assign_and_pd():
    ms = MasterScale()
    g = ms.assign([0.0005, 0.001, 0.00100001, 0.5, np.nan])
    assert list(g[:4]) == ["1", "1", "2", "12"] and g[4] is None
    gp = ms.pd_of(ms.labels)
    assert np.all(np.diff(gp) > 0)
    for lo, hi, p in zip([0.0005] + ms.upper[:-2], ms.upper[:-1], gp[:-1]):
        assert lo <= p <= hi


def test_master_scale_invalid():
    with pytest.raises(ValueError):
        MasterScale(labels=["A", "B"], upper=[0.5, 0.9])
