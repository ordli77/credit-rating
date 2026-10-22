"""End-to-End-Beispiel: Entwicklung, Out-of-Time-Validierung, Scoring, Overrides, Ablage."""
import pandas as pd

from kmu_rating import (CalibrationConfig, KMURatingModel, ModelConfig, apply_overrides,
                        generate_kmu_data, load_model, render_report, save_model)

# 1) Daten (hier synthetisch – in der Bank: Jahresabschlussdaten + Ausfallkennzeichen nach Art. 178 CRR)
df = generate_kmu_data(20000, seed=1)
dev = df[df.beobachtungsjahr <= 2022]   # Entwicklung
oot = df[df.beobachtungsjahr == 2023]   # Out-of-Time-Validierung

# 2) Konfiguration: Central Tendency (LRADR) und Margin of Conservatism
cfg = ModelConfig(calibration=CalibrationConfig(central_tendency=0.025, moc=0.05, pd_floor=0.0005))

# 3) Schätzung
model = KMURatingModel(cfg).fit(dev)
print(model.summary())

# 4) Validierung und Bericht
res_dev = model.validate(dev)
res_oot = model.validate(oot, reference=dev)
render_report(model, {"Entwicklung 2019–2022": res_dev, "Out-of-Time 2023": res_oot}, "modellbericht.html")
print(f"Gini OOT: {res_oot['gini']:.3f} ({res_oot['gini_ampel']}), PSI: {res_oot['psi_rating']:.3f}")

# 5) Ablage (versioniert, mit Prüfsumme) und Produktivscoring
save_model(model, "modell_v1")
prod = load_model("modell_v1")
ergebnis = prod.score(oot.drop(columns=["default_12m"]).head(10))
print(ergebnis[["kunden_id", "score", "pd", "ratingklasse", "hauptrisikotreiber"]].to_string())

# 6) Override durch die Kreditanalyse (dokumentiert, mit Stufenbegrenzung)
ov = pd.DataFrame([{"kunden_id": ergebnis.kunden_id.iloc[0], "override_klasse": "8",
                    "begruendung": "Verlust Hauptkunde nach Bilanzstichtag", "bearbeiter": "Kreditanalyse/MA17"}])
final, protokoll = apply_overrides(ergebnis, ov, prod.master_scale, max_upgrade=2)
print(protokoll.to_string())
