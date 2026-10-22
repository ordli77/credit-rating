"""Datenschema, Plausibilitätsprüfungen und Kennzahlenbildung (HGB-Jahresabschluss).

Eingabe ist eine Zeile je Kreditnehmer und Beobachtungszeitpunkt mit Rohpositionen
aus Bilanz und GuV (in EUR) sowie optionalen Kontodaten und qualitativen Angaben.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Pflichtfelder aus dem Jahresabschluss
REQUIRED_RAW = [
    "bilanzsumme",
    "eigenkapital",
    "verbindlichkeiten_kurzfristig",
    "verbindlichkeiten_lul",
    "bankverbindlichkeiten",
    "liquide_mittel",
    "forderungen_lul",
    "umsatz",
    "ebit",
    "zinsaufwand",
    "jahresueberschuss",
    "abschreibungen",
    "personalaufwand",
]

# Optionale Felder, die unverändert als Kandidatenmerkmale übernommen werden
OPTIONAL_PASSTHROUGH = [
    "unternehmensalter_jahre",
    "ueberziehungstage_12m",       # Kontoführung: Tage mit Überziehung in 12 Monaten
    "kontokorrent_auslastung",     # durchschnittliche Linienauslastung (0..>1)
    "branche",
    "rechtsform",
]

RATIO_DEFINITIONS = {
    "eigenkapitalquote": "Eigenkapital / Bilanzsumme",
    "bankverschuldungsquote": "Bankverbindlichkeiten / Bilanzsumme",
    "cashflow_fk_quote": "(Jahresüberschuss + Abschreibungen) / Fremdkapital",
    "zinsdeckung": "EBIT / Zinsaufwand (gekappt auf [-100, 100])",
    "umsatzrendite": "Jahresüberschuss / Umsatz",
    "gesamtkapitalrendite": "(Jahresüberschuss + Zinsaufwand) / Bilanzsumme",
    "liquiditaet_2": "(Liquide Mittel + Forderungen LuL) / kurzfr. Verbindlichkeiten",
    "kapitalumschlag": "Umsatz / Bilanzsumme",
    "kreditorenziel_tage": "Verbindlichkeiten LuL / Umsatz * 365",
    "personalaufwandsquote": "Personalaufwand / Umsatz",
    "log_umsatz": "ln(Umsatz)",
}


class DataValidationError(ValueError):
    """Pflichtfelder fehlen oder Daten sind nicht verarbeitbar."""


def _safe_div(num, den) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(num.shape, np.nan)
    ok = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[ok] = num[ok] / den[ok]
    return out


def check_schema(df: pd.DataFrame) -> None:
    """Bricht ab, wenn Pflichtfelder fehlen."""
    missing = [c for c in REQUIRED_RAW if c not in df.columns]
    if missing:
        raise DataValidationError(f"Pflichtfelder fehlen: {missing}")


def plausibility_checks(df: pd.DataFrame) -> pd.Series:
    """Harte Plausibilitätsprüfungen je Zeile.

    Rückgabe: Series mit 'OK' oder einer Fehlerbeschreibung. Zeilen mit Fehlern
    erhalten im Scoring kein Rating (PD = NaN) und gehen in die manuelle Prüfung.
    """
    check_schema(df)
    msgs = pd.Series([""] * len(df), index=df.index, dtype=object)

    def flag(mask, text):
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
        msgs.loc[mask] = msgs.loc[mask] + text + "; "

    for c in REQUIRED_RAW:
        flag(df[c].isna(), f"{c} fehlt")
    flag(df["bilanzsumme"] <= 0, "Bilanzsumme <= 0")
    flag(df["umsatz"] < 0, "Umsatz negativ")
    flag(df["eigenkapital"] > df["bilanzsumme"] * 1.0001, "Eigenkapital > Bilanzsumme")
    for c in ["liquide_mittel", "forderungen_lul", "verbindlichkeiten_kurzfristig",
              "verbindlichkeiten_lul", "bankverbindlichkeiten", "zinsaufwand",
              "personalaufwand", "abschreibungen"]:
        flag(df[c] < 0, f"{c} negativ")
    flag(
        (df["liquide_mittel"] + df["forderungen_lul"]) > df["bilanzsumme"] * 1.0001,
        "Liquide Mittel + Forderungen > Bilanzsumme",
    )
    out = msgs.str.rstrip("; ")
    out[out == ""] = "OK"
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bildet die Kandidatenkennzahlen aus den Rohpositionen.

    Nicht definierte Quotienten (Division durch 0) werden NaN und im Binning
    als eigene Klasse 'fehlend' bzw. konservativ behandelt.
    """
    check_schema(df)
    d = {c: df[c].astype(float).to_numpy() for c in REQUIRED_RAW}
    fremdkapital = d["bilanzsumme"] - d["eigenkapital"]
    cashflow = d["jahresueberschuss"] + d["abschreibungen"]

    out = pd.DataFrame(index=df.index)
    out["eigenkapitalquote"] = _safe_div(d["eigenkapital"], d["bilanzsumme"])
    out["bankverschuldungsquote"] = _safe_div(d["bankverbindlichkeiten"], d["bilanzsumme"])
    out["cashflow_fk_quote"] = _safe_div(cashflow, np.where(fremdkapital > 0, fremdkapital, np.nan))

    zd = _safe_div(d["ebit"], d["zinsaufwand"])
    no_interest = d["zinsaufwand"] == 0
    zd = np.where(no_interest & (d["ebit"] > 0), 100.0, zd)
    zd = np.where(no_interest & (d["ebit"] <= 0), 0.0, zd)
    out["zinsdeckung"] = np.clip(zd, -100.0, 100.0)

    umsatz_pos = np.where(d["umsatz"] > 0, d["umsatz"], np.nan)
    out["umsatzrendite"] = _safe_div(d["jahresueberschuss"], umsatz_pos)
    out["gesamtkapitalrendite"] = _safe_div(d["jahresueberschuss"] + d["zinsaufwand"], d["bilanzsumme"])

    liq = _safe_div(d["liquide_mittel"] + d["forderungen_lul"], d["verbindlichkeiten_kurzfristig"])
    liq = np.where(d["verbindlichkeiten_kurzfristig"] == 0, 10.0, liq)
    out["liquiditaet_2"] = np.clip(liq, 0.0, 10.0)

    out["kapitalumschlag"] = _safe_div(d["umsatz"], d["bilanzsumme"])
    out["kreditorenziel_tage"] = np.clip(_safe_div(d["verbindlichkeiten_lul"], umsatz_pos) * 365, 0, 730)
    out["personalaufwandsquote"] = _safe_div(d["personalaufwand"], umsatz_pos)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_umsatz"] = np.log(umsatz_pos)

    for c in OPTIONAL_PASSTHROUGH:
        if c in df.columns:
            out[c] = df[c]
    return out
