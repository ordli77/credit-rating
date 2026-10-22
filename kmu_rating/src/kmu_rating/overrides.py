"""Überschreibungen (Overrides) des maschinellen Ratings mit Prüfprotokoll.

Art. 172(3) CRR / MaRisk: Overrides sind zu begründen, zu dokumentieren und
regelmäßig auszuwerten. Verbesserungen über `max_upgrade` Stufen werden abgelehnt
und erfordern eine Eskalation (Zweitvotum).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .calibration import MasterScale

REQUIRED = ["kunden_id", "override_klasse", "begruendung", "bearbeiter"]


def apply_overrides(scored: pd.DataFrame, overrides: pd.DataFrame, master_scale: MasterScale,
                    max_upgrade: int = 2, id_column: str = "kunden_id") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wendet Overrides an. Rückgabe: (Ergebnis mit 'finale_klasse', Prüfprotokoll)."""
    missing = [c for c in REQUIRED if c not in overrides.columns]
    if missing:
        raise ValueError(f"Override-Datei: Spalten fehlen {missing}")
    out = scored.copy()
    out["finale_klasse"] = out["ratingklasse"]
    out["override"] = False
    pos = {l: i for i, l in enumerate(master_scale.labels)}
    log = []
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lookup = out.reset_index().set_index(id_column)["index"]

    for _, o in overrides.iterrows():
        kid, neu = o["kunden_id"], str(o["override_klasse"])
        entry = {"kunden_id": kid, "neu": neu, "bearbeiter": o["bearbeiter"],
                 "begruendung": o["begruendung"], "zeitpunkt": ts}
        if kid not in lookup.index:
            log.append({**entry, "alt": None, "status": "abgelehnt: Kunde nicht im Scoring"})
            continue
        idx = lookup[kid]
        alt = out.at[idx, "ratingklasse"]
        entry["alt"] = alt
        if neu not in pos or alt not in pos:
            log.append({**entry, "status": "abgelehnt: unbekannte Ratingklasse"})
            continue
        if not str(o["begruendung"]).strip():
            log.append({**entry, "status": "abgelehnt: Begründung fehlt"})
            continue
        stufen = pos[alt] - pos[neu]  # >0 = Verbesserung
        if stufen > max_upgrade:
            log.append({**entry, "stufen": stufen, "status": f"abgelehnt: Verbesserung > {max_upgrade} Stufen, Eskalation"})
            continue
        out.at[idx, "finale_klasse"] = neu
        out.at[idx, "override"] = True
        log.append({**entry, "stufen": stufen, "status": "angewendet"})

    out["finale_klassen_pd"] = master_scale.pd_of(out["finale_klasse"])
    return out, pd.DataFrame(log)


def override_statistics(log: pd.DataFrame) -> dict:
    ok = log[log["status"] == "angewendet"]
    if ok.empty:
        return {"anzahl": 0}
    return {"anzahl": int(len(ok)), "verbesserungen": int((ok["stufen"] > 0).sum()),
            "verschlechterungen": int((ok["stufen"] < 0).sum()),
            "mittlere_stufen": float(ok["stufen"].mean())}
