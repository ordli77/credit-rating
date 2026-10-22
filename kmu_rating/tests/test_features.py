import numpy as np
import pandas as pd
import pytest

from kmu_rating.features import (REQUIRED_RAW, DataValidationError, compute_features,
                                 plausibility_checks)


def _row(**kw):
    base = dict(bilanzsumme=1000.0, eigenkapital=300.0, verbindlichkeiten_kurzfristig=200.0,
                verbindlichkeiten_lul=100.0, bankverbindlichkeiten=400.0, liquide_mittel=50.0,
                forderungen_lul=150.0, umsatz=2000.0, ebit=100.0, zinsaufwand=20.0,
                jahresueberschuss=60.0, abschreibungen=40.0, personalaufwand=500.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_ratios_values():
    f = compute_features(_row()).iloc[0]
    assert f.eigenkapitalquote == pytest.approx(0.3)
    assert f.bankverschuldungsquote == pytest.approx(0.4)
    assert f.cashflow_fk_quote == pytest.approx(100 / 700)
    assert f.zinsdeckung == pytest.approx(5.0)
    assert f.umsatzrendite == pytest.approx(0.03)
    assert f.gesamtkapitalrendite == pytest.approx(0.08)
    assert f.liquiditaet_2 == pytest.approx(1.0)
    assert f.kapitalumschlag == pytest.approx(2.0)
    assert f.kreditorenziel_tage == pytest.approx(18.25)
    assert f.log_umsatz == pytest.approx(np.log(2000))


def test_zero_interest_and_zero_sales():
    f = compute_features(_row(zinsaufwand=0.0)).iloc[0]
    assert f.zinsdeckung == 100.0
    f = compute_features(_row(zinsaufwand=0.0, ebit=-5.0)).iloc[0]
    assert f.zinsdeckung == 0.0
    f = compute_features(_row(umsatz=0.0)).iloc[0]
    assert np.isnan(f.umsatzrendite) and np.isnan(f.log_umsatz)


def test_negative_equity_cashflow_quote_defined():
    f = compute_features(_row(eigenkapital=-100.0)).iloc[0]
    assert f.eigenkapitalquote == pytest.approx(-0.1)
    assert f.cashflow_fk_quote == pytest.approx(100 / 1100)


def test_missing_columns_raise():
    with pytest.raises(DataValidationError):
        compute_features(_row().drop(columns=["umsatz"]))


def test_plausibility():
    df = pd.concat([_row(), _row(bilanzsumme=0.0), _row(eigenkapital=2000.0), _row(umsatz=np.nan)],
                   ignore_index=True)
    s = plausibility_checks(df)
    assert s.iloc[0] == "OK"
    assert "Bilanzsumme <= 0" in s.iloc[1]
    assert "Eigenkapital > Bilanzsumme" in s.iloc[2]
    assert "umsatz fehlt" in s.iloc[3]
    assert len(REQUIRED_RAW) == 13
