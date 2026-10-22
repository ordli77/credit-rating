"""KMU-Ratingmodell: WoE-Logit-Scorecard mit Kalibrierung und Masterskala."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import __version__
from .binning import FeatureBinning, fit_binnings
from .calibration import MasterScale, apply_moc_and_floor, expit, intercept_shift
from .config import ModelConfig
from .features import compute_features, plausibility_checks
from .selection import correlation_filter, fit_logit, iv_filter, stepwise, vif, vif_filter
from . import validation as val

log = logging.getLogger("kmu_rating")


def _fmt(v) -> str:
    try:
        if pd.isna(v):
            return "fehlend"
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, float, np.number)):
        return f"{v:.3g}"
    return str(v)


class NotFittedError(RuntimeError):
    pass


def data_fingerprint(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


class KMURatingModel:
    """PD-Ratingmodell für KMU.

    Ablauf `fit`: Plausibilisierung -> Kennzahlen -> WoE-Binning -> IV-Filter ->
    Korrelationsfilter -> schrittweise Logit-Regression -> VIF-Prüfung ->
    Endschätzung -> Kalibrierung auf Central Tendency.
    """

    def __init__(self, config: ModelConfig | None = None, master_scale: MasterScale | None = None):
        self.config = config or ModelConfig()
        self.master_scale = master_scale or MasterScale()
        self.binnings: dict[str, FeatureBinning] = {}
        self.features_: list[str] = []
        self.coef_: pd.Series | None = None
        self.intercept_: float | None = None
        self.calibration_shift_: float = 0.0
        self.stats_: pd.DataFrame | None = None
        self.selection_report_: pd.DataFrame | None = None
        self.selection_log_: list[str] = []
        self.metadata_: dict = {}

    # ------------------------------------------------------------------ helpers
    def _check_fitted(self):
        if self.coef_ is None:
            raise NotFittedError("Modell ist nicht geschätzt (fit aufrufen).")

    def _woe_matrix(self, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        return pd.DataFrame({f: self.binnings[f].transform(X[f]) for f in features}, index=X.index)

    def _prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        status = plausibility_checks(df)
        X = compute_features(df)
        return X, status

    # ---------------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame, drop_invalid: bool = False) -> "KMURatingModel":
        cfg = self.config
        if cfg.target not in df.columns:
            raise ValueError(f"Zielvariable '{cfg.target}' fehlt")
        y_all = df[cfg.target]
        if not set(pd.unique(y_all.dropna())) <= {0, 1}:
            raise ValueError("Zielvariable muss binär (0/1) sein")

        X, status = self._prepare(df)
        invalid = (status != "OK") | y_all.isna()
        if invalid.any():
            if not drop_invalid:
                raise ValueError(
                    f"{int(invalid.sum())} Zeilen mit Datenfehlern; prüfen oder drop_invalid=True setzen. "
                    f"Beispiel: {status[status != 'OK'].head(3).to_dict()}")
            log.warning("%d ungültige Zeilen verworfen", int(invalid.sum()))
        X = X.loc[~invalid]
        y = y_all.loc[~invalid].astype(int).to_numpy()
        if y.sum() < 50:
            log.warning("Nur %d Ausfälle – Schätzung statistisch unsicher", int(y.sum()))

        candidates = cfg.candidate_features or [c for c in X.columns]
        candidates = [c for c in candidates if c in X.columns]
        categorical = [c for c in cfg.categorical_features if c in candidates]

        # 1) Binning
        self.binnings = fit_binnings(X, y, candidates, categorical, cfg.binning)
        ivs = {f: b.iv for f, b in self.binnings.items()}

        # 2) IV- und Korrelationsfilter
        keep, report = iv_filter(ivs, cfg.selection)
        W = self._woe_matrix(X, candidates)
        ranked = sorted(keep, key=lambda f: -ivs[f])
        after_corr, corr_drop = correlation_filter(W, ranked, cfg.selection)

        # 3) schrittweise Regression und VIF
        selected, steplog = stepwise(W, y, after_corr, cfg.selection)
        selected, viflog = vif_filter(W, y, selected, cfg.selection)
        if not selected:
            raise RuntimeError("Kein Merkmal erfüllt die Auswahlkriterien.")

        # 4) Endschätzung
        res = fit_logit(W[selected], y)
        if res is None:
            raise RuntimeError("Endschätzung konvergiert nicht.")
        if np.any(res.params[selected] >= 0):
            log.warning("Endmodell enthält Koeffizienten mit unplausiblem Vorzeichen")

        self.features_ = selected
        self.intercept_ = float(res.params["const"])
        self.coef_ = res.params[selected].astype(float)
        vifs = vif(W[selected])
        self.stats_ = pd.DataFrame({
            "koeffizient": res.params, "std_fehler": res.bse, "z": res.tvalues, "p_wert": res.pvalues,
        })
        self.stats_["vif"] = vifs.reindex(self.stats_.index)
        self.stats_["iv"] = pd.Series(ivs).reindex(self.stats_.index)

        # Auswahlprotokoll
        report["status_korrelation"] = report["merkmal"].map(corr_drop).fillna("")
        report["im_modell"] = report["merkmal"].isin(selected)
        self.selection_report_ = report
        self.selection_log_ = steplog + viflog

        # 5) Kalibrierung auf Central Tendency
        raw_logit = self.intercept_ + W[selected].to_numpy() @ self.coef_.to_numpy()
        ct = cfg.calibration.central_tendency
        self.calibration_shift_ = intercept_shift(raw_logit, ct) if ct is not None else 0.0

        pd_dev = self.predict_pd_from_woe(W[selected])
        self.metadata_ = {
            "paket_version": __version__,
            "geschaetzt_am": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "daten_fingerprint": data_fingerprint(df),
            "n_beobachtungen": int(len(y)),
            "n_ausfaelle": int(y.sum()),
            "ausfallrate_entwicklung": float(y.mean()),
            "central_tendency": ct if ct is not None else float(y.mean()),
            "gini_entwicklung": float(val.gini(y, pd_dev)),
            "merkmale": list(selected),
        }
        log.info("Modell geschätzt: %d Merkmale, Gini %.3f", len(selected), self.metadata_["gini_entwicklung"])
        return self

    # --------------------------------------------------------------- recalibrate
    def recalibrate(self, central_tendency: float, df: pd.DataFrame | None = None) -> "KMURatingModel":
        """Neue Central Tendency setzen (z. B. jährliche LRADR-Aktualisierung).

        `df` ist das Portfolio, auf dem die mittlere PD die Central Tendency treffen soll
        (Standard: Entwicklungsportfolio ist nicht gespeichert -> df erforderlich).
        """
        self._check_fitted()
        if df is None:
            raise ValueError("Kalibrierungsportfolio df erforderlich")
        X = compute_features(df)
        W = self._woe_matrix(X, self.features_)
        raw = self.intercept_ + W.to_numpy() @ self.coef_.to_numpy()
        self.calibration_shift_ = intercept_shift(raw, central_tendency)
        self.config.calibration.central_tendency = central_tendency
        self.metadata_["central_tendency"] = central_tendency
        self.metadata_["rekalibriert_am"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    # ----------------------------------------------------------------- predict
    def _logit_from_woe(self, W: pd.DataFrame) -> np.ndarray:
        return self.intercept_ + self.calibration_shift_ + W[self.features_].to_numpy() @ self.coef_.to_numpy()

    def predict_pd_from_woe(self, W: pd.DataFrame) -> np.ndarray:
        c = self.config.calibration
        return apply_moc_and_floor(expit(self._logit_from_woe(W)), c.moc, c.pd_floor)

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        """Finale PD je Kreditnehmer (kalibriert, inkl. MoC und PD-Floor). Datenfehler -> NaN."""
        self._check_fitted()
        X, status = self._prepare(df)
        p = self.predict_pd_from_woe(self._woe_matrix(X, self.features_))
        p[(status != "OK").to_numpy()] = np.nan
        return p

    @property
    def score_factor(self) -> float:
        return self.config.scaling.pdo / np.log(2)

    @property
    def score_offset(self) -> float:
        s = self.config.scaling
        return s.base_score - self.score_factor * np.log(s.base_odds)

    def _points_matrix(self, W: pd.DataFrame) -> pd.DataFrame:
        """Punkte je Merkmal: Score = Offset + Faktor * ln(Good/Bad) = Summe der Punkte."""
        m = len(self.features_)
        const = (self.score_offset - self.score_factor * (self.intercept_ + self.calibration_shift_)) / m
        return pd.DataFrame(
            {f: const - self.score_factor * self.coef_[f] * W[f].to_numpy() for f in self.features_},
            index=W.index)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vollständiges Scoring-Ergebnis für die Produktion."""
        self._check_fitted()
        X, status = self._prepare(df)
        W = self._woe_matrix(X, self.features_)
        pts = self._points_matrix(W)
        pd_final = self.predict_pd_from_woe(W)
        bad = (status != "OK").to_numpy()
        pd_final[bad] = np.nan

        out = pd.DataFrame(index=df.index)
        if self.config.id_column and self.config.id_column in df.columns:
            out[self.config.id_column] = df[self.config.id_column]
        out["score"] = np.where(bad, np.nan, pts.sum(axis=1).round(0))
        out["pd"] = pd_final
        out["ratingklasse"] = self.master_scale.assign(pd_final)
        out["klassen_pd"] = self.master_scale.pd_of(out["ratingklasse"])
        out["datenqualitaet"] = status
        reasons = self.reason_codes(df, top_n=3, _cache=(X, W, pts))
        out["hauptrisikotreiber"] = np.where(bad, "", reasons)
        out["modell_version"] = self.metadata_.get("paket_version", "")
        return out

    def reason_codes(self, df: pd.DataFrame, top_n: int = 3, _cache=None) -> np.ndarray:
        """Merkmale mit dem größten Punkteverlust ggü. der bestmöglichen Klasse."""
        self._check_fitted()
        if _cache is None:
            X = compute_features(df)
            W = self._woe_matrix(X, self.features_)
            pts = self._points_matrix(W)
        else:
            X, W, pts = _cache
        best = {f: -self.score_factor * self.coef_[f] * self.binnings[f].woe.max() for f in self.features_}
        const = (self.score_offset - self.score_factor * (self.intercept_ + self.calibration_shift_)) / len(self.features_)
        loss = pd.DataFrame({f: (const + best[f]) - pts[f] for f in self.features_}, index=pts.index)
        out = []
        for i in range(len(loss)):
            row = loss.iloc[i].sort_values(ascending=False)
            row = row[row > 0.5].head(top_n)
            out.append("; ".join(f"{f} ({_fmt(X[f].iloc[i])} → -{v:.0f} Pkt.)" for f, v in row.items()))
        return np.array(out, dtype=object)

    # ---------------------------------------------------------------- reports
    def scorecard(self) -> pd.DataFrame:
        """Scorecard: Punkte je Merkmalsklasse (gerundet)."""
        self._check_fitted()
        m = len(self.features_)
        const = (self.score_offset - self.score_factor * (self.intercept_ + self.calibration_shift_)) / m
        rows = []
        for f in self.features_:
            b = self.binnings[f]
            tab = b.table.copy()
            tab["punkte"] = (const - self.score_factor * self.coef_[f] * tab["woe"]).round(0)
            rows.append(tab)
            if not b.missing_own_bin and b.kind == "numeric":
                rows.append(pd.DataFrame([{
                    "merkmal": f, "klasse": "fehlend (konservativ)", "anzahl": 0, "ausfaelle": 0,
                    "ausfallrate": np.nan, "anteil": 0.0, "woe": b.missing_woe, "iv": 0.0,
                    "punkte": round(const - self.score_factor * self.coef_[f] * b.missing_woe)}]))
        return pd.concat(rows, ignore_index=True)

    def validate(self, df: pd.DataFrame, reference: pd.DataFrame | None = None) -> dict:
        """Validierung auf einem Datensatz mit beobachteter Zielvariable.

        `reference`: Entwicklungsdaten für Stabilitätsanalysen (PSI/CSI).
        """
        self._check_fitted()
        y_col = self.config.target
        sc = self.score(df)
        ok = sc["pd"].notna() & df[y_col].notna()
        y = df.loc[ok, y_col].astype(int).to_numpy()
        p = sc.loc[ok, "pd"].to_numpy()
        g = sc.loc[ok, "ratingklasse"].to_numpy()
        labels = self.master_scale.labels

        a, lo, hi = val.auc_confidence_interval(y, p)
        hl, hl_p = val.hosmer_lemeshow(y, p)
        hhi, hhi_norm = val.herfindahl(g, labels)
        res = {
            "n": int(len(y)), "ausfaelle": int(y.sum()), "ausfallrate": float(y.mean()),
            "mittlere_pd": float(p.mean()),
            "auc": a, "auc_ki_unten": lo, "auc_ki_oben": hi, "gini": 2 * a - 1,
            "gini_ampel": val.ampel_high_good(2 * a - 1, *val.THRESHOLDS["gini"]),
            "ks": val.ks_statistic(y, p), "brier": val.brier_score(y, p),
            "hosmer_lemeshow": hl, "hosmer_lemeshow_p": hl_p,
            "hhi": hhi, "hhi_normiert": hhi_norm,
            "hhi_ampel": val.ampel_low_good(hhi_norm, *val.THRESHOLDS["hhi"]),
            "kalibrierung_je_klasse": val.calibration_by_grade(y, g, self.master_scale.pd_of(g), labels),
            "nicht_geratet": int((~ok).sum()),
            "roc": val.roc_curve(y, p),
        }
        if reference is not None:
            sc_ref = self.score(reference)
            res["psi_rating"] = val.psi(sc_ref["ratingklasse"].dropna(), sc["ratingklasse"].dropna(), labels)
            res["psi_ampel"] = val.ampel_low_good(res["psi_rating"], *val.THRESHOLDS["psi"])
            Xr, Xc = compute_features(reference), compute_features(df)
            res["csi"] = pd.DataFrame([{
                "merkmal": f,
                "csi": val.psi(self.binnings[f].bin_labels(Xr[f]), self.binnings[f].bin_labels(Xc[f])),
            } for f in self.features_])
            res["csi"]["ampel"] = [val.ampel_low_good(v, *val.THRESHOLDS["psi"]) for v in res["csi"]["csi"]]
        return res

    def summary(self) -> str:
        self._check_fitted()
        lines = [f"KMU-Ratingmodell v{self.metadata_.get('paket_version')}",
                 f"Merkmale ({len(self.features_)}): {', '.join(self.features_)}",
                 f"Central Tendency: {self.metadata_.get('central_tendency'):.4%}",
                 f"Gini (Entwicklung): {self.metadata_.get('gini_entwicklung'):.3f}", "",
                 self.stats_.round(4).to_string()]
        return "\n".join(lines)
