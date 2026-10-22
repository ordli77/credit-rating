"""Synthetische KMU-Daten für Demo, Tests und Schulung.

ACHTUNG: Nur zu Demonstrationszwecken. Ein produktives Modell ist auf
institutseigenen Daten mit Ausfalldefinition gemäß Art. 178 CRR zu entwickeln.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BRANCHEN = {"Bau": 0.45, "Gastgewerbe": 0.55, "Handel": 0.10, "Produktion": -0.05,
            "Dienstleistung": -0.10, "Transport": 0.25, "IT": -0.20, "Gesundheit": -0.35}
RECHTSFORMEN = {"GmbH": 0.0, "GmbH & Co. KG": -0.10, "Einzelunternehmen": 0.20, "UG": 0.45, "AG": -0.20}


def generate_kmu_data(n: int = 20000, default_rate: float = 0.025, seed: int = 0,
                      years=(2019, 2020, 2021, 2022, 2023)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)  # latente Bonität (hoch = gut)

    branche = rng.choice(list(BRANCHEN), size=n, p=[.14, .06, .2, .18, .2, .07, .08, .07])
    rechtsform = rng.choice(list(RECHTSFORMEN), size=n, p=[.55, .15, .15, .1, .05])
    alter = np.round(np.clip(rng.lognormal(2.5 + 0.15 * z, 0.7), 0.5, 120), 1)

    umsatz = np.exp(rng.normal(np.log(4e6) + 0.2 * z, 1.0)).clip(2e5, 5e7)
    kapitalumschlag = np.exp(rng.normal(np.log(1.6), 0.4, n))
    bilanzsumme = umsatz / kapitalumschlag

    ekq = np.clip(0.28 + 0.12 * z + rng.normal(0, 0.10, n), -0.4, 0.9)
    eigenkapital = ekq * bilanzsumme
    fk = bilanzsumme - eigenkapital
    bank_anteil = np.clip(rng.beta(2, 3, n) - 0.08 * z, 0.0, 0.95)
    bankverb = bank_anteil * fk
    verb_lul = np.clip(rng.beta(2, 6, n) - 0.03 * z, 0.01, 0.6) * (fk - bankverb)
    vk_kurz = verb_lul + 0.35 * bankverb + 0.1 * (fk - bankverb - verb_lul)

    ebit_marge = 0.05 + 0.035 * z + rng.normal(0, 0.03, n)
    ebit = ebit_marge * umsatz
    zinsaufwand = np.maximum(bankverb * rng.uniform(0.025, 0.065, n), 0)
    jue = (ebit - zinsaufwand) * np.where(ebit - zinsaufwand > 0, 0.7, 1.0)
    abschreibungen = bilanzsumme * rng.uniform(0.03, 0.08, n)
    personal = umsatz * np.clip(rng.normal(0.30, 0.08, n) - 0.02 * z, 0.05, 0.7)

    liq = np.clip(rng.lognormal(np.log(0.08) + 0.25 * z, 0.6), 0.002, 0.4) * bilanzsumme
    ford = np.clip(rng.normal(0.15, 0.05, n), 0.01, 0.5) * bilanzsumme

    ueberz = rng.poisson(np.exp(1.0 - 1.0 * z)).astype(float)
    ueberz = np.minimum(ueberz, 365)
    kk_ausl = np.clip(rng.beta(2, 3, n) - 0.12 * z, 0, 1.3)

    lin = (-1.15 * z + 0.15 * ueberz ** 0.5 + np.vectorize(BRANCHEN.get)(branche)
           + np.vectorize(RECHTSFORMEN.get)(rechtsform) - 0.15 * np.log(alter) + rng.normal(0, 0.4, n))
    from scipy.optimize import brentq
    from scipy.special import expit
    a = brentq(lambda c: expit(c + lin).mean() - default_rate, -20, 10)
    y = rng.binomial(1, expit(a + lin))

    df = pd.DataFrame({
        "kunden_id": [f"K{i:07d}" for i in range(n)],
        "beobachtungsjahr": rng.choice(years, size=n),
        "bilanzsumme": bilanzsumme.round(0), "eigenkapital": eigenkapital.round(0),
        "verbindlichkeiten_kurzfristig": vk_kurz.round(0), "verbindlichkeiten_lul": verb_lul.round(0),
        "bankverbindlichkeiten": bankverb.round(0), "liquide_mittel": liq.round(0),
        "forderungen_lul": ford.round(0), "umsatz": umsatz.round(0), "ebit": ebit.round(0),
        "zinsaufwand": zinsaufwand.round(0), "jahresueberschuss": jue.round(0),
        "abschreibungen": abschreibungen.round(0), "personalaufwand": personal.round(0),
        "unternehmensalter_jahre": alter, "ueberziehungstage_12m": ueberz,
        "kontokorrent_auslastung": kk_ausl.round(3), "branche": branche, "rechtsform": rechtsform,
        "default_12m": y,
    })
    # realistische Datenlücken in Kontodaten (z. B. Neukunden ohne Kontohistorie)
    miss = rng.random(n) < 0.06
    df.loc[miss, ["ueberziehungstage_12m", "kontokorrent_auslastung"]] = np.nan
    return df
