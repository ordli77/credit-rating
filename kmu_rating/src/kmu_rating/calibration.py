"""Kalibrierung und Masterskala."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit


def intercept_shift(logits: np.ndarray, target_rate: float) -> float:
    """Verschiebung d, sodass mean(expit(logits + d)) = target_rate (Central Tendency)."""
    if not 0 < target_rate < 1:
        raise ValueError("Zielausfallrate muss in (0, 1) liegen")
    f = lambda d: float(np.mean(expit(logits + d))) - target_rate
    return float(brentq(f, -30.0, 30.0))


def apply_moc_and_floor(pd_cal: np.ndarray, moc: float, floor: float) -> np.ndarray:
    return np.clip(np.asarray(pd_cal) * (1.0 + moc), floor, 1.0)


@dataclass
class MasterScale:
    """Masterskala: Ratingklassen mit PD-Obergrenzen und Klassen-PD.

    `upper` ist die Obergrenze (inklusive) der Klasse; die letzte Klasse muss 1.0 haben.
    `grade_pd` ist die der Klasse zugewiesene PD (Standard: geometrisches Mittel der Grenzen).
    """

    labels: list = field(default_factory=lambda: [str(i) for i in range(1, 13)])
    upper: list = field(default_factory=lambda: [
        0.0010, 0.0020, 0.0035, 0.0060, 0.0100, 0.0160,
        0.0260, 0.0420, 0.0700, 0.1200, 0.2000, 1.0])
    grade_pd: list | None = None
    last_grade_pd: float = 0.30

    def __post_init__(self):
        if len(self.labels) != len(self.upper):
            raise ValueError("labels und upper müssen gleich lang sein")
        if any(np.diff(self.upper) <= 0) or self.upper[-1] != 1.0:
            raise ValueError("upper muss streng steigend sein und mit 1.0 enden")
        if self.grade_pd is None:
            lower = [0.0005] + list(self.upper[:-1])
            gp = [float(np.sqrt(lo * hi)) for lo, hi in zip(lower[:-1], self.upper[:-1])]
            self.grade_pd = gp + [self.last_grade_pd]

    def assign(self, pd_values) -> np.ndarray:
        p = np.asarray(pd_values, dtype=float)
        idx = np.searchsorted(self.upper, p, side="left")
        idx = np.clip(idx, 0, len(self.labels) - 1)
        out = np.array(self.labels, dtype=object)[idx]
        out[~np.isfinite(p)] = None
        return out

    def pd_of(self, grades) -> np.ndarray:
        m = dict(zip(self.labels, self.grade_pd))
        return np.array([m.get(g, np.nan) for g in grades], dtype=float)

    def index_of(self, grades) -> np.ndarray:
        m = {l: i for i, l in enumerate(self.labels)}
        return np.array([m.get(g, -1) for g in grades], dtype=int)

    def table(self) -> pd.DataFrame:
        lower = [0.0] + list(self.upper[:-1])
        return pd.DataFrame({"klasse": self.labels, "pd_von": lower,
                             "pd_bis": self.upper, "klassen_pd": self.grade_pd})


__all__ = ["intercept_shift", "apply_moc_and_floor", "MasterScale", "expit", "logit"]
