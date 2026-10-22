"""HTML-Modell- und Validierungsbericht (eine Datei, ohne externe Abhängigkeiten)."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .features import RATIO_DEFINITIONS
from .model import KMURatingModel

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#666;--line:#ddd;--head:#f3f4f6;--acc:#1f4e79}
@media (prefers-color-scheme: dark){:root{--bg:#16181b;--fg:#e8e8e8;--muted:#aaa;--line:#333;--head:#22262b;--acc:#7fb3e6}}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);
max-width:1100px;margin:0 auto;padding:24px;line-height:1.45}
h1{color:var(--acc)}h2{border-bottom:2px solid var(--acc);padding-bottom:4px;margin-top:36px}
table{border-collapse:collapse;font-size:13px;margin:10px 0;width:100%}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:right}
th{background:var(--head)}td:first-child,th:first-child{text-align:left}
.wrap{overflow-x:auto}.muted{color:var(--muted)}
.grün{background:#2e7d32;color:#fff}.gelb{background:#f9a825;color:#000}.rot{background:#c62828;color:#fff}
.kpi{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:8px 14px;margin:4px}
"""


def _table(df: pd.DataFrame, pct_cols=(), fmt=None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["<div class='wrap'><table><tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols) + "</tr>"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cls = ""
            if c == "ampel" or str(c).endswith("_ampel"):
                cls = f" class='{v}'"
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    s = ""
                elif c in pct_cols:
                    s = f"{v:.2%}"
                else:
                    s = format(v, fmt.get(c, ".4f"))
            else:
                s = html.escape(str(v))
            cells.append(f"<td{cls}>{s}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(out) + "</table></div>"


def _roc_svg(curves: dict) -> str:
    w = 320
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    parts = [f"<svg viewBox='0 0 {w + 40} {w + 40}' width='360' role='img' aria-label='ROC-Kurve'>",
             f"<rect x='30' y='10' width='{w}' height='{w}' fill='none' stroke='currentColor' stroke-opacity='.4'/>",
             f"<line x1='30' y1='{w + 10}' x2='{w + 30}' y2='10' stroke='currentColor' stroke-dasharray='4' stroke-opacity='.4'/>"]
    for i, (name, (fpr, tpr)) in enumerate(curves.items()):
        step = max(1, len(fpr) // 400)
        pts = " ".join(f"{30 + x * w:.1f},{10 + (1 - y) * w:.1f}" for x, y in zip(fpr[::step], tpr[::step]))
        pts += f" {30 + w},{10}"
        parts.append(f"<polyline fill='none' stroke='{colors[i % 3]}' stroke-width='2' points='{pts}'/>")
        parts.append(f"<text x='{w - 90}' y='{w - 10 - 16 * i}' font-size='12' fill='{colors[i % 3]}'>{html.escape(name)}</text>")
    parts.append(f"<text x='{w / 2}' y='{w + 36}' font-size='11' fill='currentColor'>FPR</text>")
    parts.append(f"<text x='0' y='{w / 2}' font-size='11' fill='currentColor'>TPR</text></svg>")
    return "".join(parts)


def _val_section(name: str, r: dict) -> str:
    kpis = [("N", f"{r['n']:,}".replace(",", ".")), ("Ausfälle", r["ausfaelle"]),
            ("Ausfallrate", f"{r['ausfallrate']:.2%}"), ("Ø PD", f"{r['mittlere_pd']:.2%}"),
            ("Gini", f"{r['gini']:.3f}"), ("AUC 95 %-KI", f"[{r['auc_ki_unten']:.3f}; {r['auc_ki_oben']:.3f}]"),
            ("KS", f"{r['ks']:.3f}"), ("Brier", f"{r['brier']:.4f}"),
            ("HL p-Wert", f"{r['hosmer_lemeshow_p']:.3f}"), ("HHI*", f"{r['hhi_normiert']:.3f}")]
    s = [f"<h3>{html.escape(name)}</h3>",
         "".join(f"<span class='kpi'><b>{k}</b><br>{v}</span>" for k, v in kpis),
         f"<p>Ampel Trennschärfe: <span class='{r['gini_ampel']}'>&nbsp;{r['gini_ampel']}&nbsp;</span> · "
         f"Ampel Konzentration: <span class='{r['hhi_ampel']}'>&nbsp;{r['hhi_ampel']}&nbsp;</span>"]
    if "psi_rating" in r:
        s.append(f" · PSI Ratingverteilung {r['psi_rating']:.3f} <span class='{r['psi_ampel']}'>&nbsp;{r['psi_ampel']}&nbsp;</span>")
    s.append("</p><h4>Kalibrierung je Ratingklasse (Binomial- und Jeffreys-Test, einseitig)</h4>")
    s.append(_table(r["kalibrierung_je_klasse"], pct_cols=("ausfallrate", "klassen_pd")))
    if "csi" in r:
        s.append("<h4>Merkmalsstabilität (CSI)</h4>" + _table(r["csi"]))
    return "\n".join(s)


def render_report(model: KMURatingModel, results: dict[str, dict], path: str | Path,
                  title: str = "KMU-Ratingmodell – Modell- und Validierungsbericht") -> Path:
    """`results`: {Name der Stichprobe: Ergebnis von model.validate(...)}."""
    m = model.metadata_
    defs = pd.DataFrame([{"merkmal": f, "definition": RATIO_DEFINITIONS.get(f, "Rohmerkmal / qualitativ")}
                         for f in model.features_])
    stats = model.stats_.reset_index().rename(columns={"index": "parameter"})
    body = [
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='muted'>Erstellt {datetime.now():%d.%m.%Y %H:%M} · Paketversion {m.get('paket_version')} · "
        f"geschätzt {m.get('geschaetzt_am')} · Daten-Fingerprint {m.get('daten_fingerprint', '')[:16]}…</p>",
        "<h2>1. Modellüberblick</h2>",
        f"<p>Logistische Regression auf WoE-transformierten Merkmalen; {m.get('n_beobachtungen')} Beobachtungen, "
        f"{m.get('n_ausfaelle')} Ausfälle (Ausfallrate {m.get('ausfallrate_entwicklung', 0):.2%}). "
        f"Kalibriert auf Central Tendency {m.get('central_tendency', 0):.2%}; MoC {model.config.calibration.moc:.0%}; "
        f"PD-Floor {model.config.calibration.pd_floor:.2%}.</p>",
        _table(defs),
        "<h2>2. Merkmalsauswahl</h2>",
        _table(model.selection_report_, fmt={"iv": ".4f"}),
        "<p class='muted'>Protokoll: " + html.escape(" | ".join(model.selection_log_) or "–") + "</p>",
        "<h2>3. Koeffizienten</h2>",
        _table(stats),
        "<h2>4. Scorecard</h2>",
        f"<p class='muted'>Skalierung: {model.config.scaling.base_score:.0f} Punkte bei Odds "
        f"{model.config.scaling.base_odds:.0f}:1, PDO {model.config.scaling.pdo:.0f}.</p>",
        _table(model.scorecard(), pct_cols=("ausfallrate", "anteil"), fmt={"punkte": ".0f"}),
        "<h2>5. Masterskala</h2>",
        _table(model.master_scale.table(), pct_cols=("pd_von", "pd_bis", "klassen_pd")),
        "<h2>6. Validierung</h2>",
        _roc_svg({k: v["roc"] for k, v in results.items()}),
    ]
    for name, r in results.items():
        body.append(_val_section(name, r))
    body.append("<h2>7. Hinweise</h2><p class='muted'>Ampelschwellen sind Vorschlagswerte und gemäß "
                "Validierungsrichtlinie des Instituts festzulegen. Der Bericht ersetzt nicht die "
                "unabhängige Validierung.</p>")
    doc = (f"<!DOCTYPE html><html lang='de'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
           + "\n".join(body) + "</body></html>")
    p = Path(path)
    p.write_text(doc, encoding="utf-8")
    return p
