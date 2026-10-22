"""Kommandozeile: kmu-rating {demo-daten,train,score,validate}."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import ModelConfig
from .data import generate_kmu_data
from .model import KMURatingModel
from .persistence import load_model, save_model
from .report import render_report


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, sep=None, engine="python")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kmu-rating", description="KMU-PD-Ratingmodell")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo-daten", help="synthetische Beispieldaten erzeugen")
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, default=20000)
    d.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train", help="Modell schätzen")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True, help="Zielverzeichnis für das Modell")
    t.add_argument("--config", help="JSON-Konfiguration (ModelConfig)")
    t.add_argument("--drop-invalid", action="store_true")
    t.add_argument("--report", help="HTML-Bericht (Entwicklungsstichprobe)")

    s = sub.add_parser("score", help="Kreditnehmer raten")
    s.add_argument("--model", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)

    v = sub.add_parser("validate", help="Validierung + HTML-Bericht")
    v.add_argument("--model", required=True)
    v.add_argument("--data", required=True)
    v.add_argument("--reference", help="Entwicklungsdaten für PSI/CSI")
    v.add_argument("--report", required=True)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.cmd == "demo-daten":
        generate_kmu_data(a.n, seed=a.seed).to_csv(a.out, index=False)
    elif a.cmd == "train":
        cfg = ModelConfig.from_dict(json.loads(Path(a.config).read_text())) if a.config else ModelConfig()
        df = _read(a.data)
        m = KMURatingModel(cfg).fit(df, drop_invalid=a.drop_invalid)
        save_model(m, a.out)
        print(m.summary())
        if a.report:
            render_report(m, {"Entwicklung": m.validate(df.dropna(subset=[cfg.target]))}, a.report)
    elif a.cmd == "score":
        m = load_model(a.model)
        m.score(_read(a.data)).to_csv(a.out, index=False)
    elif a.cmd == "validate":
        m = load_model(a.model)
        df = _read(a.data)
        ref = _read(a.reference) if a.reference else None
        render_report(m, {"Validierung": m.validate(df, reference=ref)}, a.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
