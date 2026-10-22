"""Validierungskennzahlen (Trennschärfe, Kalibrierung, Stabilität, Konzentration).

Orientierung: EBA/GL/2017/16, EZB-Leitfaden zu internen Modellen und
"Instructions for reporting the validation results of internal models" (Jeffreys-Test).
Die Ampelschwellen sind marktübliche Vorschläge und von der Bank in der
Validierungsrichtlinie festzulegen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, chi2, rankdata

# Ampelschwellen (Vorschlag)
THRESHOLDS = {
    "gini": (0.50, 0.40),        # grün >= 0.50, gelb >= 0.40, sonst rot
    "psi": (0.10, 0.25),         # grün < 0.10, gelb < 0.25, sonst rot
    "p_value": (0.05, 0.01),     # grün >= 5 %, gelb >= 1 %, sonst rot
    "hhi": (0.20, 0.30),         # grün < 0.20, gelb < 0.30 (normierter HHI)
}


def ampel_high_good(value: float, green: float, yellow: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return "grün" if value >= green else ("gelb" if value >= yellow else "rot")


def ampel_low_good(value: float, green: float, yellow: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return "grün" if value < green else ("gelb" if value < yellow else "rot")


# ---------------------------------------------------------------- Trennschärfe
def auc(y, risk) -> float:
    """AUC = P(Risiko eines Ausfalls > Risiko eines Nicht-Ausfalls); Bindungen zählen 1/2."""
    y = np.asarray(y, int)
    r = rankdata(np.asarray(risk, float))
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_confidence_interval(y, risk, level: float = 0.95) -> tuple[float, float, float]:
    """AUC mit Konfidenzintervall nach Hanley & McNeil (1982)."""
    a = auc(y, risk)
    y = np.asarray(y, int)
    n1, n0 = y.sum(), len(y) - y.sum()
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))
    from scipy.stats import norm
    z = norm.ppf(0.5 + level / 2)
    return a, max(0.0, a - z * se), min(1.0, a + z * se)


def gini(y, risk) -> float:
    return 2 * auc(y, risk) - 1


def ks_statistic(y, risk) -> float:
    y = np.asarray(y, int)
    risk = np.asarray(risk, float)
    order = np.argsort(-risk, kind="mergesort")
    ys = y[order]
    cb = np.cumsum(ys) / max(ys.sum(), 1)
    cg = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    # nur an Stellen mit Wertwechsel auswerten (Bindungen)
    rs = risk[order]
    last = np.r_[rs[1:] != rs[:-1], True]
    return float(np.max(np.abs(cb[last] - cg[last])))


def roc_curve(y, risk) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, int)
    risk = np.asarray(risk, float)
    order = np.argsort(-risk, kind="mergesort")
    ys, rs = y[order], risk[order]
    last = np.r_[rs[1:] != rs[:-1], True]
    tpr = np.r_[0, (np.cumsum(ys) / ys.sum())[last]]
    fpr = np.r_[0, (np.cumsum(1 - ys) / (1 - ys).sum())[last]]
    return fpr, tpr


# ------------------------------------------------------------------ Kalibrierung
def brier_score(y, pd_) -> float:
    return float(np.mean((np.asarray(pd_, float) - np.asarray(y, float)) ** 2))


def hosmer_lemeshow(y, pd_, groups: int = 10) -> tuple[float, float]:
    """HL-Statistik über PD-Quantilsgruppen; df = Gruppen - 2."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(pd_, float)})
    df["g"] = pd.qcut(df["p"].rank(method="first"), groups, labels=False)
    agg = df.groupby("g").agg(n=("y", "size"), d=("y", "sum"), e=("p", "sum"))
    pbar = agg["e"] / agg["n"]
    stat = float((((agg["d"] - agg["e"]) ** 2) / (agg["n"] * pbar * (1 - pbar))).sum())
    dof = max(len(agg) - 2, 1)
    return stat, float(chi2.sf(stat, dof))


def binomial_test(n: int, d: int, pd_: float) -> float:
    """Einseitig: H0 PD korrekt vs. H1 PD unterschätzt. p = P(X >= d)."""
    return float(binom.sf(d - 1, n, pd_)) if n > 0 else np.nan


def jeffreys_test(n: int, d: int, pd_: float) -> float:
    """Jeffreys-Test (EZB): p = F_Beta(d+1/2, n-d+1/2)(PD). Kleiner Wert -> PD zu niedrig."""
    return float(beta.cdf(pd_, d + 0.5, n - d + 0.5)) if n > 0 else np.nan


def calibration_by_grade(y, grades, grade_pd, labels: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, float), "g": grades, "p": np.asarray(grade_pd, float)})
    rows = []
    for lab in labels:
        sub = df[df["g"] == lab]
        n, d = len(sub), int(sub["y"].sum())
        if n == 0:
            continue
        p = float(sub["p"].mean())
        pj = jeffreys_test(n, d, p)
        rows.append({
            "klasse": lab, "anzahl": n, "ausfaelle": d, "ausfallrate": d / n, "klassen_pd": p,
            "p_binomial": binomial_test(n, d, p), "p_jeffreys": pj,
            "ampel": ampel_high_good(pj, *THRESHOLDS["p_value"]),
        })
    tot_n, tot_d = len(df), int(df["y"].sum())
    p_tot = float(df["p"].mean())
    pj = jeffreys_test(tot_n, tot_d, p_tot)
    rows.append({"klasse": "Portfolio", "anzahl": tot_n, "ausfaelle": tot_d,
                 "ausfallrate": tot_d / max(tot_n, 1), "klassen_pd": p_tot,
                 "p_binomial": binomial_test(tot_n, tot_d, p_tot), "p_jeffreys": pj,
                 "ampel": ampel_high_good(pj, *THRESHOLDS["p_value"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- Stabilität
def psi(expected, actual, categories=None, eps: float = 1e-4) -> float:
    """Population Stability Index für kategoriale Verteilungen (z. B. Ratingklassen, WoE-Klassen)."""
    e = pd.Series(expected).astype(str)
    a = pd.Series(actual).astype(str)
    cats = categories if categories is not None else sorted(set(e) | set(a))
    pe = e.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + eps
    pa = a.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy() + eps
    pe, pa = pe / pe.sum(), pa / pa.sum()
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def herfindahl(grades, labels: list[str]) -> tuple[float, float]:
    """HHI der Klassenbesetzung und normierter HHI* = (HHI - 1/K) / (1 - 1/K)."""
    share = pd.Series(grades).value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
    h = float(np.sum(share ** 2))
    k = len(labels)
    return h, (h - 1 / k) / (1 - 1 / k)


def migration_matrix(old, new, labels: list[str]) -> pd.DataFrame:
    """Migrationsmatrix (Zeilen: alt, Spalten: neu), relativ je Zeile."""
    tab = pd.crosstab(pd.Categorical(old, labels), pd.Categorical(new, labels), dropna=False)
    tab.index.name, tab.columns.name = "alt", "neu"
    return tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0)


def migration_stats(old, new, labels: list[str]) -> dict:
    idx = {l: i for i, l in enumerate(labels)}
    o = np.array([idx.get(x, -1) for x in old])
    n = np.array([idx.get(x, -1) for x in new])
    ok = (o >= 0) & (n >= 0)
    diff = n[ok] - o[ok]
    return {"stabil": float(np.mean(diff == 0)), "innerhalb_1_stufe": float(np.mean(np.abs(diff) <= 1)),
            "verschlechtert": float(np.mean(diff > 0)), "verbessert": float(np.mean(diff < 0))}
