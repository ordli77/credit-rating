"""Merkmalsauswahl: IV-Filter, Korrelationsfilter, schrittweise Regression, VIF."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import SelectionConfig


def fit_logit(X: pd.DataFrame, y):
    """Logit-Schätzung (ML) mit statsmodels; gibt None bei Nichtkonvergenz zurück."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.Logit(np.asarray(y, float), sm.add_constant(X, has_constant="add")).fit(
                disp=0, method="newton", maxiter=100)
        except Exception:
            return None
    if not res.mle_retvals.get("converged", False) or not np.all(np.isfinite(res.bse)):
        return None
    return res


def _signs_ok(res, cols) -> bool:
    # WoE-Konvention: höheres WoE = besser -> Koeffizient muss negativ sein
    return bool(np.all(res.params[cols] < 0))


def iv_filter(ivs: dict[str, float], cfg: SelectionConfig) -> tuple[list[str], pd.DataFrame]:
    rows = []
    for f, iv in sorted(ivs.items(), key=lambda kv: -kv[1]):
        if iv < cfg.min_iv:
            status = "verworfen: IV zu gering"
        elif iv > cfg.max_iv:
            status = "verworfen: IV verdächtig hoch (Leakage prüfen)"
        else:
            status = "ok"
        rows.append({"merkmal": f, "iv": iv, "status_iv": status})
    tab = pd.DataFrame(rows)
    keep = tab.loc[tab["status_iv"] == "ok", "merkmal"].tolist()
    forced = [f for f in cfg.forced_features if f in ivs and f not in keep]
    return keep + forced, tab


def correlation_filter(W: pd.DataFrame, ranked: list[str], cfg: SelectionConfig) -> tuple[list[str], dict]:
    """Greedy: Merkmale nach IV absteigend; verwirft Merkmale mit zu hoher Korrelation."""
    corr = W[ranked].corr().abs()
    selected, dropped = [], {}
    for f in ranked:
        clash = [s for s in selected if corr.loc[f, s] > cfg.max_corr]
        if clash and f not in cfg.forced_features:
            dropped[f] = f"verworfen: Korrelation {corr.loc[f, clash[0]]:.2f} mit {clash[0]}"
        else:
            selected.append(f)
    return selected, dropped


def stepwise(W: pd.DataFrame, y, candidates: list[str], cfg: SelectionConfig) -> tuple[list[str], list[str]]:
    """Schrittweise Vorwärtsauswahl mit Rückwärtsprüfung.

    Aufnahme nur, wenn p < p_enter und alle Koeffizienten das erwartete
    (negative) Vorzeichen haben. Auswahlkriterium: größter Likelihood-Gewinn.
    """
    log = []
    selected = [f for f in cfg.forced_features if f in candidates]
    remaining = [c for c in candidates if c not in selected]
    base = fit_logit(W[selected], y) if selected else fit_logit(pd.DataFrame(index=W.index), y)
    base_llf = base.llf if base is not None else -np.inf

    for _ in range(4 * len(candidates) + 1):
        if len(selected) >= cfg.max_features or not remaining:
            break
        best, best_llf = None, base_llf
        for c in remaining:
            res = fit_logit(W[selected + [c]], y)
            if res is None or res.pvalues[c] >= cfg.p_enter or not _signs_ok(res, selected + [c]):
                continue
            if res.llf > best_llf:
                best, best_llf = c, res.llf
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
        base_llf = best_llf
        log.append(f"aufgenommen: {best}")

        res = fit_logit(W[selected], y)
        while res is not None:
            removable = [s for s in selected if s not in cfg.forced_features]
            if not removable:
                break
            worst = max(removable, key=lambda s: res.pvalues[s])
            if res.pvalues[worst] <= cfg.p_remove:
                break
            selected.remove(worst)
            log.append(f"entfernt (p={res.pvalues[worst]:.3f}): {worst}")
            res = fit_logit(W[selected], y)
            base_llf = res.llf if res is not None else -np.inf
    return selected, log


def vif(W: pd.DataFrame) -> pd.Series:
    """Varianzinflationsfaktoren (über Inverse der Korrelationsmatrix)."""
    if W.shape[1] < 2:
        return pd.Series(1.0, index=W.columns)
    c = np.corrcoef(W.to_numpy().T)
    inv = np.linalg.pinv(c)
    return pd.Series(np.diag(inv), index=W.columns)


def vif_filter(W: pd.DataFrame, y, selected: list[str], cfg: SelectionConfig) -> tuple[list[str], list[str]]:
    log = []
    sel = list(selected)
    while len(sel) > 1:
        v = vif(W[sel])
        cand = v.drop([f for f in cfg.forced_features if f in v.index], errors="ignore")
        if cand.empty or cand.max() <= cfg.max_vif:
            break
        worst = cand.idxmax()
        sel.remove(worst)
        log.append(f"entfernt (VIF={v[worst]:.2f}): {worst}")
    return sel, log
