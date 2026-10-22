"""Konfiguration des Ratingmodells.

Alle methodischen Stellschrauben sind hier zentral und versionierbar abgelegt,
damit die Modellentwicklung reproduzierbar und für Validierung/Revision
nachvollziehbar ist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class BinningConfig:
    """Parameter des Grob-Klassierens (WoE-Binning)."""

    n_initial_bins: int = 20          # Feinklassierung über Quantile
    max_bins: int = 6                 # maximale Anzahl Klassen je Merkmal
    min_bin_share: float = 0.05       # Mindestanteil Beobachtungen je Klasse
    monotonic: bool = True            # monotone Ausfallraten erzwingen
    monotonic_overrides: dict = field(default_factory=dict)  # {merkmal: bool}
    woe_smoothing: float = 0.5        # Laplace-Glättung bei WoE-Berechnung


@dataclass
class SelectionConfig:
    """Parameter der Merkmalsauswahl."""

    min_iv: float = 0.02              # Information Value Untergrenze
    max_iv: float = 1.0               # IV darüber -> Verdacht auf Target Leakage
    max_corr: float = 0.6             # max. |Pearson-Korrelation| der WoE-Merkmale
    max_vif: float = 5.0              # max. Varianzinflationsfaktor
    p_enter: float = 0.05             # Aufnahme-Signifikanz (Forward)
    p_remove: float = 0.10            # Entfernungs-Signifikanz (Backward)
    max_features: int = 12            # max. Anzahl Merkmale im Endmodell
    forced_features: list = field(default_factory=list)  # fachlich gesetzte Merkmale


@dataclass
class ScalingConfig:
    """Skalierung der Scorepunkte: Score = Offset + Faktor * ln(Good/Bad-Odds)."""

    base_score: float = 600.0
    base_odds: float = 50.0           # Good:Bad-Odds beim base_score
    pdo: float = 20.0                 # Points to Double the Odds


@dataclass
class CalibrationConfig:
    """Kalibrierung auf die langfristige Ausfallrate (Central Tendency)."""

    central_tendency: Optional[float] = None  # LRADR; None = Ausfallrate Entwicklungsstichprobe
    moc: float = 0.0                  # Margin of Conservatism, relativer Aufschlag (0.1 = +10 %)
    pd_floor: float = 0.0005          # PD-Untergrenze (CRR3 Art. 160(1): 0,05 %)


@dataclass
class ModelConfig:
    """Gesamtkonfiguration."""

    target: str = "default_12m"
    id_column: Optional[str] = "kunden_id"
    candidate_features: Optional[list] = None   # None = alle verfügbaren Kennzahlen
    categorical_features: list = field(default_factory=lambda: ["branche", "rechtsform"])
    binning: BinningConfig = field(default_factory=BinningConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    random_state: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        return cls(
            binning=BinningConfig(**d.pop("binning", {})),
            selection=SelectionConfig(**d.pop("selection", {})),
            scaling=ScalingConfig(**d.pop("scaling", {})),
            calibration=CalibrationConfig(**d.pop("calibration", {})),
            **d,
        )
