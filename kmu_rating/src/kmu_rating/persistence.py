"""Speichern/Laden mit Integritätsprüfung (SHA-256) und Metadaten.

Hinweis: joblib/pickle führt beim Laden Code aus. Nur Modelldateien aus der
eigenen, zugriffsgeschützten Modellablage laden.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from . import __version__
from .model import KMURatingModel


class IntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(model: KMURatingModel, directory: str | Path) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    mpath = d / "model.joblib"
    joblib.dump(model, mpath)
    meta = {
        **model.metadata_,
        "config": model.config.to_dict(),
        "masterskala": model.master_scale.table().to_dict(orient="list"),
        "koeffizienten": {"const": model.intercept_, **model.coef_.to_dict()},
        "kalibrierungs_shift": model.calibration_shift_,
        "sha256_model": _sha256(mpath),
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    model.scorecard().to_csv(d / "scorecard.csv", index=False, sep=";", decimal=",")
    return d


def load_model(directory: str | Path, verify: bool = True) -> KMURatingModel:
    d = Path(directory)
    mpath = d / "model.joblib"
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    if verify and _sha256(mpath) != meta["sha256_model"]:
        raise IntegrityError("Prüfsumme stimmt nicht – Modelldatei wurde verändert.")
    if meta.get("paket_version", "").split(".")[0] != __version__.split(".")[0]:
        raise IntegrityError(f"Modell mit Paketversion {meta.get('paket_version')} erstellt, "
                             f"installiert ist {__version__} (Major-Version abweichend).")
    model = joblib.load(mpath)
    if not isinstance(model, KMURatingModel):
        raise IntegrityError("Datei enthält kein KMURatingModel")
    return model
