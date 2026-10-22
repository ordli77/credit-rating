import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, binom

from kmu_rating import validation as v


def test_auc_perfect_random_ties():
    y = np.array([0, 0, 1, 1])
    assert v.auc(y, [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert v.auc(y, [0.4, 0.3, 0.2, 0.1]) == 0.0
    assert v.auc(y, [0.5] * 4) == 0.5
    assert v.gini(y, [0.1, 0.2, 0.3, 0.4]) == 1.0


def test_auc_matches_pairwise():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 400)
    s = rng.normal(size=400) + y
    s[:50] = np.round(s[:50])
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert v.auc(y, s) == pytest.approx(brute)
    a, lo, hi = v.auc_confidence_interval(y, s)
    assert lo < a < hi


def test_ks_perfect():
    assert v.ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_binomial_and_jeffreys():
    assert v.binomial_test(100, 5, 0.02) == pytest.approx(binom.sf(4, 100, 0.02))
    assert v.jeffreys_test(100, 5, 0.02) == pytest.approx(beta.cdf(0.02, 5.5, 95.5))
    assert v.jeffreys_test(1000, 60, 0.02) < 0.001     # PD klar unterschätzt


def test_hosmer_lemeshow_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.2, 20000)
    y = rng.binomial(1, p)
    _, pv = v.hosmer_lemeshow(y, p)
    assert pv > 0.01
    _, pv_bad = v.hosmer_lemeshow(y, p * 0.5)
    assert pv_bad < 1e-6


def test_psi_hhi_migration():
    assert v.psi(list("aabb"), list("aabb")) == pytest.approx(0, abs=1e-12)
    assert v.psi(list("a" * 90 + "b" * 10), list("a" * 50 + "b" * 50)) > 0.25
    h, hn = v.herfindahl(["1"] * 10, ["1", "2", "3"])
    assert h == 1 and hn == pytest.approx(1)
    mm = v.migration_matrix(["1", "1", "2"], ["1", "2", "2"], ["1", "2"])
    assert mm.loc["1", "2"] == 0.5 and mm.loc["2", "2"] == 1.0
    st = v.migration_stats(["1", "1", "2"], ["1", "2", "2"], ["1", "2"])
    assert st["stabil"] == pytest.approx(2 / 3)
