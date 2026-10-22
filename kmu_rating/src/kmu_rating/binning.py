"""Grobklassierung (Coarse Classing) und Weight of Evidence.

Konvention: WoE = ln( Anteil Gute / Anteil Schlechte ).
Hoher WoE = geringes Risiko. In der PD-Logit-Regression müssen die
Koeffizienten daher negativ sein (Plausibilitätsprüfung im Modell).

Behandlung fehlender / unbekannter Werte:
- Fehlende Werte mit ausreichender Fallzahl bilden eine eigene Klasse.
- Sonst (und bei im Training unbekannten Kategorien) wird konservativ der
  WoE der risikoreichsten Klasse vergeben.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import BinningConfig

MISSING_LABEL = "fehlend"
OTHER_LABEL = "__SONSTIGE__"


def _chi2_pair(n1, b1, n2, b2) -> float:
    g1, g2 = n1 - b1, n2 - b2
    n = n1 + n2
    tb, tg = b1 + b2, g1 + g2
    if n == 0 or tb == 0 or tg == 0:
        return 0.0
    chi = 0.0
    for obs, rown, coltot in ((b1, n1, tb), (g1, n1, tg), (b2, n2, tb), (g2, n2, tg)):
        exp = rown * coltot / n
        if exp > 0:
            chi += (obs - exp) ** 2 / exp
    return chi


def _rate(n, b):
    return b / n if n > 0 else 0.0


class FeatureBinning:
    """Binning eines einzelnen Merkmals (numerisch oder kategorial)."""

    def __init__(self, name: str, kind: str, config: BinningConfig, monotonic: bool | None = None):
        if kind not in ("numeric", "categorical"):
            raise ValueError("kind muss 'numeric' oder 'categorical' sein")
        self.name = name
        self.kind = kind
        self.config = config
        self.monotonic = config.monotonic if monotonic is None else monotonic
        self.cuts: list[float] = []
        self.mapping: dict = {}          # Kategorie -> Bin-Index
        self.group_labels: list[str] = []
        self.woe: np.ndarray = np.array([])
        self.missing_woe: float = 0.0
        self.missing_own_bin = False
        self.unseen_woe: float = 0.0
        self.table: pd.DataFrame | None = None
        self.iv: float = 0.0
        self.direction: int = 0

    # ------------------------------------------------------------------ fit
    def fit(self, x: pd.Series, y) -> "FeatureBinning":
        y = np.asarray(y, dtype=float)
        if self.kind == "numeric":
            n, b, miss_n, miss_b = self._fit_numeric(pd.to_numeric(x, errors="coerce"), y)
        else:
            n, b, miss_n, miss_b = self._fit_categorical(x, y)
        self._finalize(np.asarray(n, float), np.asarray(b, float), miss_n, miss_b, len(y))
        return self

    def _min_count(self, n_total: int) -> int:
        return max(int(math.ceil(self.config.min_bin_share * n_total)), 1)

    def _merge_loop(self, n: list, b: list, merge_fn, min_count: int, ordered: bool):
        """Gemeinsame Merge-Logik: Mindestgröße, Monotonie, max. Klassenanzahl."""
        # 1) zu kleine oder reine Klassen (nur Gute / nur Schlechte) auflösen
        while len(n) > 1:
            bad = [i for i in range(len(n)) if n[i] < min_count or b[i] == 0 or b[i] == n[i]]
            if not bad:
                break
            i = min(bad, key=lambda k: n[k])
            if i == 0:
                j = 0
            elif i == len(n) - 1:
                j = i - 1
            else:
                r = _rate(n[i], b[i])
                left, right = abs(r - _rate(n[i - 1], b[i - 1])), abs(r - _rate(n[i + 1], b[i + 1]))
                j = i - 1 if left <= right else i
            merge_fn(j)
        # 2) Monotonie (nur für geordnete Merkmale)
        if ordered and self.monotonic and len(n) > 1:
            while True:
                rates = [_rate(n[i], b[i]) for i in range(len(n))]
                viol = [i for i in range(len(n) - 1) if self.direction * (rates[i + 1] - rates[i]) < 0]
                if not viol:
                    break
                merge_fn(viol[0])
        # 3) maximale Klassenanzahl
        while len(n) > self.config.max_bins:
            chis = [_chi2_pair(n[i], b[i], n[i + 1], b[i + 1]) for i in range(len(n) - 1)]
            merge_fn(int(np.argmin(chis)))

    def _fit_numeric(self, x: pd.Series, y: np.ndarray):
        xv_all = x.to_numpy(dtype=float)
        mask = np.isfinite(xv_all)
        xv, yv = xv_all[mask], y[mask]
        min_count = self._min_count(len(y))
        if len(xv) == 0:
            raise ValueError(f"Merkmal {self.name}: keine gültigen Werte")

        qs = np.quantile(xv, np.linspace(0, 1, self.config.n_initial_bins + 1)[1:-1])
        cuts = [c for c in np.unique(qs) if c < xv.max()]
        idx = np.searchsorted(cuts, xv, side="left")
        k = len(cuts) + 1
        n = list(np.bincount(idx, minlength=k).astype(float))
        b = list(np.bincount(idx, weights=yv, minlength=k).astype(float))

        if len(np.unique(xv)) > 1:
            rho = spearmanr(xv, yv).statistic
            self.direction = 1 if (np.isnan(rho) or rho >= 0) else -1
        else:
            self.direction = 1

        def merge(i):
            n[i] += n[i + 1]
            b[i] += b[i + 1]
            del n[i + 1], b[i + 1], cuts[i]

        self._merge_loop(n, b, merge, min_count, ordered=True)
        self.cuts = [float(c) for c in cuts]
        return n, b, int((~mask).sum()), float(y[~mask].sum())

    def _fit_categorical(self, x: pd.Series, y: np.ndarray):
        xs = x.astype(object).where(x.notna(), MISSING_LABEL).astype(str)
        min_count = self._min_count(len(y))
        stats = pd.DataFrame({"x": xs.to_numpy(), "y": y}).groupby("x")["y"].agg(["count", "sum"])
        rare = stats[stats["count"] < min_count]
        groups: list[list[str]] = [[c] for c in stats.index if c not in rare.index]
        n = [float(stats.loc[g[0], "count"]) for g in groups]
        b = [float(stats.loc[g[0], "sum"]) for g in groups]
        if len(rare):
            groups.append(list(rare.index) + [OTHER_LABEL])
            n.append(float(rare["count"].sum()))
            b.append(float(rare["sum"].sum()))
        order = np.argsort([_rate(n[i], b[i]) for i in range(len(n))], kind="stable")
        groups = [groups[i] for i in order]
        n = [n[i] for i in order]
        b = [b[i] for i in order]

        def merge(i):
            n[i] += n[i + 1]
            b[i] += b[i + 1]
            groups[i] = groups[i] + groups[i + 1]
            del n[i + 1], b[i + 1], groups[i + 1]

        self.direction = 1
        self._merge_loop(n, b, merge, min_count, ordered=True)
        self.mapping = {c: gi for gi, g in enumerate(groups) for c in g}
        self.group_labels = [", ".join(sorted(c for c in g if c != OTHER_LABEL)) or "Sonstige" for g in groups]
        return n, b, 0, 0.0  # Missing ist als Kategorie enthalten

    def _finalize(self, n, b, miss_n, miss_b, n_total):
        a = self.config.woe_smoothing
        min_count = self._min_count(n_total)
        self.missing_own_bin = self.kind == "numeric" and miss_n >= min_count
        if self.missing_own_bin:
            n_all = np.append(n, miss_n)
            b_all = np.append(b, miss_b)
        else:
            n_all, b_all = n, b
        g_all = n_all - b_all
        G, B = g_all.sum() + (0 if self.missing_own_bin else miss_n - miss_b), b_all.sum() + (0 if self.missing_own_bin else miss_b)
        K = len(n_all)
        pg = (g_all + a) / (G + a * K)
        pb = (b_all + a) / (B + a * K)
        woe_all = np.log(pg / pb)
        iv_parts = (pg - pb) * woe_all

        self.woe = woe_all[: len(n)]
        worst = float(self.woe.min())
        self.missing_woe = float(woe_all[-1]) if self.missing_own_bin else worst
        self.unseen_woe = worst
        if self.kind == "categorical" and OTHER_LABEL in self.mapping:
            self.unseen_woe = float(self.woe[self.mapping[OTHER_LABEL]])
        self.iv = float(iv_parts.sum())

        labels = self._labels()
        if self.missing_own_bin:
            labels = labels + [MISSING_LABEL]
        self.table = pd.DataFrame({
            "merkmal": self.name,
            "klasse": labels,
            "anzahl": n_all.astype(int),
            "ausfaelle": b_all.astype(int),
            "ausfallrate": np.where(n_all > 0, b_all / np.maximum(n_all, 1), np.nan),
            "anteil": n_all / n_total,
            "woe": woe_all,
            "iv": iv_parts,
        })

    def _labels(self) -> list[str]:
        if self.kind == "categorical":
            return list(self.group_labels)
        edges = [-np.inf] + self.cuts + [np.inf]
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if np.isinf(lo):
                out.append(f"<= {hi:.4g}")
            elif np.isinf(hi):
                out.append(f"> {lo:.4g}")
            else:
                out.append(f"({lo:.4g}, {hi:.4g}]")
        return out

    # ------------------------------------------------------------ transform
    def bin_index(self, x: pd.Series) -> np.ndarray:
        """Klassenindex je Beobachtung; -1 = fehlend, -2 = unbekannte Kategorie."""
        if self.kind == "numeric":
            xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
            idx = np.searchsorted(self.cuts, xv, side="left").astype(int)
            idx[~np.isfinite(xv)] = -1
            return idx
        xs = x.astype(object).where(x.notna(), MISSING_LABEL).astype(str)
        return np.array([self.mapping.get(v, -2) for v in xs], dtype=int)

    def transform(self, x: pd.Series) -> np.ndarray:
        idx = self.bin_index(x)
        out = np.empty(len(idx), dtype=float)
        regular = idx >= 0
        out[regular] = self.woe[idx[regular]]
        out[idx == -1] = self.missing_woe
        out[idx == -2] = self.unseen_woe
        return out

    def bin_labels(self, x: pd.Series) -> np.ndarray:
        idx = self.bin_index(x)
        labels = np.array(self._labels(), dtype=object)
        out = np.empty(len(idx), dtype=object)
        out[idx >= 0] = labels[idx[idx >= 0]]
        out[idx == -1] = MISSING_LABEL
        out[idx == -2] = "unbekannt"
        return out


def fit_binnings(X: pd.DataFrame, y, features: list[str], categorical: list[str],
                 config: BinningConfig) -> dict[str, FeatureBinning]:
    result = {}
    for f in features:
        kind = "categorical" if f in categorical else "numeric"
        mono = config.monotonic_overrides.get(f)
        result[f] = FeatureBinning(f, kind, config, monotonic=mono).fit(X[f], y)
    return result
