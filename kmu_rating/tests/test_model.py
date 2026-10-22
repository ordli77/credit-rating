import numpy as np
import pandas as pd
import pytest

from kmu_rating import (CalibrationConfig, KMURatingModel, ModelConfig, SelectionConfig,
                        apply_overrides, load_model, render_report, save_model)
from kmu_rating.calibration import expit
from kmu_rating.model import NotFittedError
from kmu_rating.persistence import IntegrityError


def test_fit_basic_properties(model):
    assert 3 <= len(model.features_) <= 12
    assert (model.coef_ < 0).all()
    assert (model.stats_.loc[model.features_, "p_wert"] < 0.10).all()
    assert (model.stats_.loc[model.features_, "vif"] <= 5).all()
    assert model.metadata_["gini_entwicklung"] > 0.5


def test_calibration_hits_central_tendency(model, dev):
    assert model.predict_pd(dev).mean() == pytest.approx(0.03, rel=1e-3)


def test_oot_performance_and_calibration(model, oot, dev):
    r = model.validate(oot, reference=dev)
    assert r["gini"] > 0.45
    assert r["psi_rating"] < 0.1
    assert r["kalibrierung_je_klasse"].iloc[-1]["klasse"] == "Portfolio"


def test_score_consistent_with_pd(model, oot):
    sc = model.score(oot)
    # Score = Offset + Faktor * ln(Odds) der kalibrierten PD (vor MoC/Floor; hier MoC = 0)
    p = sc["pd"].to_numpy()
    expected = model.score_offset + model.score_factor * np.log((1 - p) / p)
    mask = p > model.config.calibration.pd_floor
    assert np.abs(sc["score"].to_numpy()[mask] - expected[mask]).max() <= 0.5 + 1e-6
    assert (np.diff(sc.sort_values("pd")["score"].to_numpy()) <= 0.5).all()  # monoton (Rundung)


def test_scorecard_points_sum(model, oot):
    card = model.scorecard()
    assert set(card["merkmal"]) == set(model.features_)
    assert card["punkte"].notna().all()


def test_invalid_rows_not_rated(model, oot):
    bad = oot.head(5).copy()
    bad.loc[bad.index[0], "bilanzsumme"] = -1
    sc = model.score(bad)
    assert np.isnan(sc["pd"].iloc[0]) and pd.isna(sc["ratingklasse"].iloc[0])
    assert sc["datenqualitaet"].iloc[0] != "OK"
    assert sc["pd"].iloc[1:].notna().all()


def test_fit_rejects_invalid_unless_dropped(dev):
    d = dev.copy()
    d.loc[d.index[0], "bilanzsumme"] = 0
    with pytest.raises(ValueError):
        KMURatingModel().fit(d)
    KMURatingModel().fit(d, drop_invalid=True)


def test_unseen_category_and_missing(model, oot):
    x = oot.head(3).copy()
    x["branche"] = "Raumfahrt"
    x["ueberziehungstage_12m"] = np.nan
    assert np.isfinite(model.predict_pd(x)).all()


def test_moc_and_floor():
    from kmu_rating import generate_kmu_data
    d = generate_kmu_data(8000, seed=3)
    m0 = KMURatingModel(ModelConfig(calibration=CalibrationConfig(central_tendency=0.02))).fit(d)
    m1 = KMURatingModel(ModelConfig(calibration=CalibrationConfig(central_tendency=0.02, moc=0.2,
                                                                 pd_floor=0.003))).fit(d)
    p0, p1 = m0.predict_pd(d), m1.predict_pd(d)
    assert p1.min() >= 0.003
    assert np.allclose(np.maximum(p0 * 1.2, 0.003)[p0 > 0.003], p1[p0 > 0.003])


def test_recalibrate(model, oot):
    m = KMURatingModel(model.config, model.master_scale)
    m.__dict__.update({k: v for k, v in model.__dict__.items() if k != "config"})
    import copy
    m.config = copy.deepcopy(model.config)
    m.recalibrate(0.05, oot)
    assert m.predict_pd(oot).mean() == pytest.approx(0.05, rel=1e-3)


def test_forced_feature(dev):
    cfg = ModelConfig(selection=SelectionConfig(forced_features=["liquiditaet_2"]))
    m = KMURatingModel(cfg).fit(dev)
    assert "liquiditaet_2" in m.features_


def test_not_fitted():
    with pytest.raises(NotFittedError):
        KMURatingModel().predict_pd(pd.DataFrame())


def test_save_load_roundtrip(model, oot, tmp_path):
    save_model(model, tmp_path / "m")
    m2 = load_model(tmp_path / "m")
    assert np.allclose(model.predict_pd(oot), m2.predict_pd(oot))
    assert (tmp_path / "m" / "scorecard.csv").exists()
    with open(tmp_path / "m" / "model.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(IntegrityError):
        load_model(tmp_path / "m")


def test_overrides(model, oot):
    sc = model.score(oot.head(10))
    ids = sc["kunden_id"].tolist()
    g0 = sc["ratingklasse"].iloc[0]
    worse = model.master_scale.labels[min(model.master_scale.labels.index(g0) + 1, 11)]
    ov = pd.DataFrame([
        {"kunden_id": ids[0], "override_klasse": worse, "begruendung": "Negativmerkmal Auskunftei", "bearbeiter": "MA1"},
        {"kunden_id": ids[1], "override_klasse": "1", "begruendung": "x", "bearbeiter": "MA1"},
        {"kunden_id": "UNBEKANNT", "override_klasse": "3", "begruendung": "x", "bearbeiter": "MA1"},
    ])
    out, logdf = apply_overrides(sc, ov, model.master_scale, max_upgrade=0)
    assert out["finale_klasse"].iloc[0] == worse and out["override"].iloc[0]
    assert logdf["status"].iloc[2].startswith("abgelehnt")
    if sc["ratingklasse"].iloc[1] != "1":
        assert logdf["status"].iloc[1].startswith("abgelehnt")


def test_report(model, oot, dev, tmp_path):
    p = render_report(model, {"OOT 2023": model.validate(oot, reference=dev)}, tmp_path / "r.html")
    txt = p.read_text(encoding="utf-8")
    assert "<svg" in txt and "Scorecard" in txt and "Jeffreys" in txt


def test_cli(tmp_path):
    from kmu_rating.cli import main
    data = tmp_path / "d.csv"
    assert main(["demo-daten", "--out", str(data), "--n", "6000"]) == 0
    assert main(["train", "--data", str(data), "--out", str(tmp_path / "m")]) == 0
    assert main(["score", "--model", str(tmp_path / "m"), "--data", str(data), "--out", str(tmp_path / "s.csv")]) == 0
    assert main(["validate", "--model", str(tmp_path / "m"), "--data", str(data),
                 "--reference", str(data), "--report", str(tmp_path / "r.html")]) == 0
    s = pd.read_csv(tmp_path / "s.csv")
    assert {"pd", "ratingklasse", "score", "hauptrisikotreiber"} <= set(s.columns)
