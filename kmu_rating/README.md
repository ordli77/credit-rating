# kmu-rating

PD-Ratingmodell für kleine und mittlere Unternehmen (KMU) auf Basis einer
logistischen Regression mit WoE-Transformation (klassische Scorecard),
inklusive Kalibrierung, Masterskala, Validierung, Override-Prozess und
revisionssicherer Modellablage.

## Installation

```bash
pip install -e ".[dev]"
pytest            # 37 Tests
```

## Schnellstart

```python
from kmu_rating import KMURatingModel, ModelConfig, CalibrationConfig, save_model, load_model

cfg = ModelConfig(calibration=CalibrationConfig(central_tendency=0.025, moc=0.05))
model = KMURatingModel(cfg).fit(entwicklungsdaten)
model.validate(oot_daten, reference=entwicklungsdaten)   # Gini, Jeffreys, PSI, ...
save_model(model, "modelle/kmu_v1")

ergebnis = load_model("modelle/kmu_v1").score(neue_kunden)
# -> score, pd, ratingklasse, klassen_pd, datenqualitaet, hauptrisikotreiber
```

Kommandozeile:

```bash
kmu-rating demo-daten --out demo.csv
kmu-rating train    --data dev.csv --out modelle/kmu_v1 --report entwicklung.html
kmu-rating score    --model modelle/kmu_v1 --data kunden.csv --out rating.csv
kmu-rating validate --model modelle/kmu_v1 --data oot.csv --reference dev.csv --report validierung.html
```

Ein vollständiger Ablauf steht in `examples/beispiel_workflow.py`.

## Eingabedaten

Eine Zeile je Kreditnehmer und Beobachtungszeitpunkt (Beträge in EUR).

| Pflichtfelder (HGB-Jahresabschluss) | |
|---|---|
| `bilanzsumme`, `eigenkapital`, `verbindlichkeiten_kurzfristig`, `verbindlichkeiten_lul`, `bankverbindlichkeiten`, `liquide_mittel`, `forderungen_lul` | Bilanz |
| `umsatz`, `ebit`, `zinsaufwand`, `jahresueberschuss`, `abschreibungen`, `personalaufwand` | GuV |

Optional: `unternehmensalter_jahre`, `ueberziehungstage_12m`, `kontokorrent_auslastung`,
`branche`, `rechtsform`, `kunden_id`. Zielvariable: `default_12m` (0/1).

Daraus werden elf Kennzahlen gebildet (Eigenkapitalquote, Bankverschuldungsquote,
Cashflow/Fremdkapital, Zinsdeckung, Umsatz- und Gesamtkapitalrendite, Liquidität 2. Grades,
Kapitalumschlag, Kreditorenziel, Personalaufwandsquote, ln Umsatz); Definitionen in
`features.RATIO_DEFINITIONS`.

Zeilen, die harte Plausibilitätsprüfungen verletzen (z. B. Bilanzsumme ≤ 0, Eigenkapital >
Bilanzsumme, fehlende Pflichtwerte), erhalten **kein** maschinelles Rating (PD = NaN,
Hinweis in `datenqualitaet`) und gehören in die manuelle Bearbeitung.

## Methodik

1. **Grobklassierung**: Quantils-Feinklassierung, dann Zusammenlegen bis Mindestgröße
   (5 %), monotone Ausfallraten (abschaltbar je Merkmal) und max. 6 Klassen (χ²-Kriterium).
   WoE = ln(Anteil Gute / Anteil Schlechte). Fehlende Werte: eigene Klasse ab Mindestgröße,
   sonst konservativ WoE der schlechtesten Klasse; ebenso unbekannte Kategorien.
2. **Merkmalsauswahl**: IV-Filter (0,02 ≤ IV ≤ 1,0; höher → Leakage-Verdacht),
   Korrelationsfilter (|ρ| ≤ 0,6), schrittweise Logit-Regression (p_enter 5 %, p_remove 10 %)
   mit Vorzeichenrestriktion, VIF ≤ 5. Fachlich gesetzte Merkmale über `forced_features`.
3. **Kalibrierung**: Intercept-Verschiebung auf die Central Tendency (langfristige
   Ausfallrate), danach relativer MoC-Aufschlag und PD-Floor (Standard 0,05 %, CRR3).
4. **Masterskala**: 12 Klassen, konfigurierbar (`MasterScale(labels, upper, grade_pd)`).
5. **Scorecard**: Score = 600 bei Odds 50:1, PDO 20; Punkte je Merkmalsklasse in
   `model.scorecard()`. Der Score entspricht der kalibrierten PD *vor* MoC und Floor.
6. **Hauptrisikotreiber**: je Kunde die drei Merkmale mit dem größten Punkteverlust
   gegenüber der besten Klasse (Begründung für Kreditentscheidung und Kundengespräch).

## Validierung

`model.validate(df, reference=None)` liefert: AUC/Gini mit Konfidenzintervall
(Hanley-McNeil), KS, Brier, Hosmer-Lemeshow, Binomial- und Jeffreys-Test je Ratingklasse
und Portfolio, Herfindahl-Index, PSI der Ratingverteilung und CSI je Merkmal, jeweils mit
Ampel. `validation.migration_matrix` und `migration_stats` für Ratingmigrationen.
`render_report` erzeugt einen eigenständigen HTML-Bericht (Merkmalsauswahl, Koeffizienten,
Scorecard, Masterskala, ROC-Kurven, Validierungsergebnisse).

Die Ampelschwellen in `validation.THRESHOLDS` sind Vorschlagswerte und gemäß
Validierungsrichtlinie des Instituts festzulegen.

## Governance-Funktionen

- **Modellablage** (`save_model`): Modell, `metadata.json` (Konfiguration, Koeffizienten,
  Masterskala, Daten-Fingerprint, Zeitstempel) und `scorecard.csv`; `load_model` prüft
  SHA-256-Prüfsumme und Major-Version. Pickle-Dateien nur aus geschützter Ablage laden.
- **Overrides** (`apply_overrides`): Begründungspflicht, Bearbeiter, Zeitstempel,
  Begrenzung von Verbesserungen (`max_upgrade`), Protokoll für die Override-Analyse.
- **Rekalibrierung** (`recalibrate`): jährliche Aktualisierung der Central Tendency.

## Was die Bank vor dem Produktiveinsatz selbst leisten muss

Das Paket ist ein vollständiges, getestetes Werkzeug für Entwicklung, Anwendung und
Validierung. Ein aufsichtlich belastbares Ratingverfahren entsteht erst mit:

- **Institutseigenen Daten** mit Ausfalldefinition nach Art. 178 CRR / EBA/GL/2016/07
  (90 Tage Verzug, Unlikeliness-to-pay), ausreichend langer Historie (IRB: mind. 5 Jahre,
  Art. 180 CRR) und Datenqualitätsprozessen. Die mitgelieferten Daten sind synthetisch.
- **Langfristiger Ausfallrate** inkl. eines wirtschaftlichen Abschwungs für die Central
  Tendency und einer begründeten MoC-Quantifizierung (EBA/GL/2017/16).
- **Fachlicher Prüfung** der ausgewählten Merkmale und Klassengrenzen durch die
  Kreditanalyse (Plausibilität, Ökonomie, Anwendbarkeit).
- **Unabhängiger Validierung**, Modellgenehmigung und Einbindung in die
  Kreditprozesse gemäß MaRisk (BTO 1.4 Risikoklassifizierungsverfahren) sowie
  Model-Risk-Management; bei IRB zusätzlich aufsichtliche Zulassung.
- **Segmentierung prüfen**: Je nach Portfolio separate Modelle oder Merkmale für
  Bilanzierer vs. Einnahmen-Überschuss-Rechner, Existenzgründer und Branchen.
- **Konservatives Behandeln veralteter Abschlüsse** (z. B. Abschlag bei Bilanz älter
  als 18 Monate) gemäß interner Richtlinie.
