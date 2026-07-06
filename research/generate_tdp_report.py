#!/usr/bin/env python3
"""
generate_tdp_report.py
======================
ASTM WKXXXXX v1.0.0 — Technical Dossier Package (TDP) Report Generator

Reads CHO results JSON file(s) produced by run_cho_analysis_v6_0.py and
generates:
  1. A structured JSON TDP with full regulatory traceability.
  2. A PDF report (ReportLab) with tier-specific sections, auto-claim strings,
     and a comparison table.

Auto-claim logic
----------------
For each tier:
  - SUPERIOR     : ΔAUC CI lower bound > 0
  - NON-INFERIOR : ΔAUC CI lower bound > −0.05 and upper bound ≥ 0
  - INDETERMINATE: otherwise

Claim string format:
  "MAR implementation '{name}' demonstrates {outcome} lesion detectability
  vs noMAR for {tier_id} ({blockage:.0f}% blockage, {material}) with
  ΔAUC = {val:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}];
  one-tailed Wilcoxon p = {p:.4f})."

Usage
-----
    python generate_tdp_report.py \\
        --results results_t1_ab.json \\
        --mar-name "AlgoName v2.1" \\
        --output-dir ./tdp_output

    # Multiple tiers in one report:
    python generate_tdp_report.py \\
        --results results_t1_ab.json results_t2_sb.json results_t3_head.json \\
        --mar-name "AlgoName v2.1"
"""

from __future__ import annotations

import argparse
import io
import json
import os
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image as RLImage,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# tier_config lives at the repo root; add it to the path so this script runs
# from research/ (matches the shim pattern used in algorithms/).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tier_config import TierConfig, TIER_REGISTRY

# ─── Font paths ───────────────────────────────────────────────────────────────
_DEJAVU_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_BOLD    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PDF_FONT        = "DejaVuSans"
PDF_FONT_BOLD   = "DejaVuSans-Bold"

STANDARD_REF = "ASTM-WKXXXXX-v1.0.0"

# ─── Script location (for relative path resolution) ──────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS_DIR = _SCRIPT_DIR / "results_archive"

# ─── Known tier ID fragments in filenames ────────────────────────────────────
_TIER_FRAGMENTS = {"t1_ab": "T1_AB", "t2_sb": "T2_SB", "t3_head": "T3_HEAD"}

# ─── Filename-to-algorithm mapping ──────────────────────────────────────────
_ALGO_FRAGMENTS = {
    "spectral":  "Spectral MAR",
    "mbir_imar": "MBIR-iMAR",
    "asd_imar":  "ASD-iMAR",
    "dlsc_imar": "DLSC-iMAR",
    "fs_imar":   "FS-iMAR",
    "pocs_tv":   "POCS-TV",
    "nmar":      "NMAR",
    "imar":      "iMAR",
}


def infer_algorithm_tier(filename: str) -> tuple[str, str]:
    """
    Infer (algorithm_name, tier_id) from a result JSON filename.

    Handles two naming conventions:
      - {algo}_{tier}_results.json   →  imar_t2_sb_results.json
      - results_{tier}_{algo}.json   →  results_t2_sb_fs_imar.json
      - results_{tier}.json          →  results_t1_ab.json  (defaults to iMAR)
    """
    stem = Path(filename).stem.lower()  # e.g. "asd_imar_t2_sb_results"

    # Determine tier
    tier_id = "UNKNOWN"
    for frag, tid in _TIER_FRAGMENTS.items():
        if frag in stem:
            tier_id = tid
            break

    # Determine algorithm — check longest fragments first to avoid partial matches
    algo_name = "iMAR"  # default
    for frag, name in _ALGO_FRAGMENTS.items():
        if frag in stem:
            algo_name = name
            break

    return algo_name, tier_id


def auto_discover_results(results_dir: Path) -> dict[str, dict[str, dict]]:
    """
    Auto-discover all result JSONs and return {tier_id: {algo_name: result_dict}}.

    Deduplicates: if both results_t2_sb.json and imar_t2_sb_results.json exist
    (same algo+tier), the more specifically named file wins.
    """
    by_tier: dict[str, dict[str, dict]] = {}
    seen: dict[tuple[str, str], str] = {}  # (algo, tier) → filename

    for p in sorted(results_dir.glob("*.json")):
        algo, tier = infer_algorithm_tier(p.name)
        if tier == "UNKNOWN":
            print(f"  Skipping (unknown tier): {p.name}")
            continue

        key = (algo, tier)
        # Prefer {algo}_{tier}_results.json over results_{tier}.json
        if key in seen:
            existing = seen[key]
            # If the new file has the algo name explicitly, prefer it
            if algo.lower().replace("-", "_") in p.stem.lower() and \
               algo.lower().replace("-", "_") not in Path(existing).stem.lower():
                print(f"  Replacing {existing} with {p.name} for {algo}/{tier}")
            else:
                print(f"  Skipping duplicate {p.name} ({algo}/{tier}, already have {existing})")
                continue

        try:
            with open(p) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  Warning: could not load {p}: {exc}")
            continue

        by_tier.setdefault(tier, {})[algo] = data
        seen[key] = p.name
        print(f"  Loaded: {p.name} → {algo} / {tier}")

    return by_tier


# ─── Claim string generation ──────────────────────────────────────────────────

def classify_outcome(delta_auc: float, ci_lo: float, ci_hi: float) -> str:
    """
    Classify the MAR performance outcome.

      SUPERIOR     : lower CI bound > 0  (positive ΔAUC with statistical confidence)
      NON-INFERIOR : lower CI bound > −0.05 and CI upper bound ≥ 0
      INDETERMINATE: lower CI bound ≤ −0.05 or upper CI bound < 0
    """
    if ci_lo > 0.0:
        return "SUPERIOR"
    if ci_lo > -0.05 and ci_hi >= 0.0:
        return "NON-INFERIOR"
    return "INDETERMINATE"


def generate_claim_string(
    mar_name: str,
    tier: TierConfig,
    delta_auc: float,
    ci_lo: float,
    ci_hi: float,
    p_wilcoxon: float,
) -> str:
    """Auto-generate the §8.5 regulatory claim string for one tier."""
    outcome = classify_outcome(delta_auc, ci_lo, ci_hi)
    return (
        f"MAR implementation '{mar_name}' demonstrates {outcome} lesion "
        f"detectability vs noMAR for {tier.tier_id} "
        f"({tier.description}; {tier.blockage_frac*100:.0f}% angular blockage, "
        f"{tier.metal_material}) with "
        f"ΔAUC = {delta_auc:+.4f} "
        f"(95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]; "
        f"one-tailed Wilcoxon p = {p_wilcoxon:.4f})."
    )


# ─── JSON TDP construction ─────────────────────────────────────────────────────

def build_tdp_json(
    results_list: list[dict],
    mar_name: str,
) -> dict:
    """
    Construct the structured Technical Dossier Package JSON.

    Parameters
    ----------
    results_list : list of CHO results dicts (one per tier)
    mar_name     : MAR algorithm name / version string
    """
    tiers_output = []
    all_claims = []

    for res in results_list:
        tier_d = res.get("tier", {})
        tier_id = tier_d.get("tier_id", "UNKNOWN")
        tier = TIER_REGISTRY.get(tier_id)

        noMAR = res.get("noMAR", {})
        MAR   = res.get("MAR", {})
        delta_auc = res.get("delta_AUC", float("nan"))
        ci = res.get("delta_AUC_CI_95", [float("nan"), float("nan")])
        ci_lo, ci_hi = float(ci[0]), float(ci[1])

        # Wilcoxon p from MAR condition
        mar_wilcoxon = MAR.get("wilcoxon", {})
        p_wilcoxon = float(mar_wilcoxon.get("p_one_sided", float("nan")))

        outcome = classify_outcome(delta_auc, ci_lo, ci_hi)
        claim = generate_claim_string(
            mar_name, tier, delta_auc, ci_lo, ci_hi, p_wilcoxon
        ) if tier else f"[Tier {tier_id} not in registry]"
        all_claims.append({"tier_id": tier_id, "outcome": outcome, "claim": claim})

        # Noise sweep sigmoid fit (from noMAR if available)
        sigmoid = None
        ns = noMAR.get("noise_sweep")
        if ns:
            sigmoid = ns.get("sigmoid_fit")

        tiers_output.append({
            "tier_id": tier_id,
            "tier_params": tier_d,
            "noMAR": {
                "AUC_hold_out":   noMAR.get("AUC"),
                "bias_rs_minus_ho": noMAR.get("Bias"),
                "CI_95":          noMAR.get("CI_95"),
                "bias_5050":      noMAR.get("bias_5050", {}).get("Bias_5050"),
                "wilcoxon_p":     noMAR.get("wilcoxon", {}).get("p_one_sided"),
                "sigmoid_fit":    sigmoid,
            },
            "MAR": {
                "AUC_hold_out":   MAR.get("AUC"),
                "bias_rs_minus_ho": MAR.get("Bias"),
                "CI_95":          MAR.get("CI_95"),
                "bias_5050":      MAR.get("bias_5050", {}).get("Bias_5050"),
                "wilcoxon_p":     p_wilcoxon,
                "significant_p05": mar_wilcoxon.get("significant_p05"),
            },
            "delta_AUC":             float(delta_auc),
            "delta_AUC_CI_95":       [ci_lo, ci_hi],
            "delta_wilcoxon_p_twosided": res.get("delta_wilcoxon_p_twosided"),
            "outcome":               outcome,
            "claim":                 claim,
            "internal_noise_sigma":  res.get("internal_noise_sigma"),
            "reconstruction_pipeline": res.get("reconstruction_pipeline"),
        })

    # Mandatory T1_AB check
    t1_ab_tiers = [t for t in tiers_output if t["tier_id"] == "T1_AB"]
    t1_ab_ok = bool(t1_ab_tiers)

    tdp = {
        "tdp_version":    "1.0.0",
        "standard":       STANDARD_REF,
        "generation_date": date.today().isoformat(),
        "generation_ts":   datetime.now(timezone.utc).isoformat(),
        "mar_implementation": mar_name,
        "t1_ab_present":  t1_ab_ok,
        "t1_ab_note": (
            "T1_AB is present (mandatory anchor tier)."
            if t1_ab_ok
            else "WARNING: T1_AB (mandatory anchor tier) is missing from this TDP."
        ),
        "tiers": tiers_output,
        "claims": all_claims,
        "traceability": {
            "observer":       "2D Channelized Hotelling Observer (CHO)",
            "channels":       "10 Laguerre-Gauss (tier-specific a)",
            "covariance":     "LA condition only (§A1.5.2(a))",
            "regularization": "Tikhonov λ = 0.01 × trace(K) / 10",
            "hold_out":       "Leave-One-Out (LOO) with observer re-fit per fold",
            "bootstrap":      "1000 resamples, observation-level, on LOO scores",
            "wilcoxon":       "One-tailed paired Wilcoxon on d[i] = s_LP[i] − s_LA[i]",
            "estimation_bias":"50/50 2-fold CV",
            "metric":         "Mann-Whitney AUC (mid-rank tie correction)",
            "note_3d":        "3D integration across z shall not be performed (§A1.5.3)",
        },
    }
    return tdp


def build_comparison_tdp(
    by_tier: dict[str, dict[str, dict]],
) -> dict:
    """
    Build a multi-algorithm comparison TDP from auto-discovered results.

    Parameters
    ----------
    by_tier : {tier_id: {algo_name: result_dict}}
    """
    comparison_tiers = []
    all_algorithms = sorted({algo for algos in by_tier.values() for algo in algos})

    for tier_id in ["T1_AB", "T2_SB", "T3_HEAD"]:
        if tier_id not in by_tier:
            continue
        algos = by_tier[tier_id]
        tier = TIER_REGISTRY.get(tier_id)

        algo_rows = []
        for algo_name in all_algorithms:
            if algo_name not in algos:
                continue
            res = algos[algo_name]
            noMAR = res.get("noMAR", {})
            MAR = res.get("MAR", {})
            delta = res.get("delta_AUC", float("nan"))
            ci = res.get("delta_AUC_CI_95", [float("nan"), float("nan")])
            ci_lo, ci_hi = float(ci[0]), float(ci[1])
            outcome = classify_outcome(delta, ci_lo, ci_hi)
            mar_wilcoxon = MAR.get("wilcoxon", {})
            p_val = float(mar_wilcoxon.get("p_one_sided", float("nan")))

            algo_rows.append({
                "algorithm":     algo_name,
                "AUC_noMAR":     noMAR.get("AUC"),
                "AUC_MAR":       MAR.get("AUC"),
                "delta_AUC":     delta,
                "delta_AUC_CI":  [ci_lo, ci_hi],
                "wilcoxon_p":    p_val,
                "outcome":       outcome,
                "claim": generate_claim_string(
                    algo_name, tier, delta, ci_lo, ci_hi, p_val
                ) if tier else "",
            })

        comparison_tiers.append({
            "tier_id":     tier_id,
            "description": tier.description if tier else tier_id,
            "algorithms":  algo_rows,
        })

    t1_present = any(t["tier_id"] == "T1_AB" for t in comparison_tiers)

    return {
        "tdp_version":    "1.0.0",
        "standard":       STANDARD_REF,
        "generation_date": date.today().isoformat(),
        "generation_ts":   datetime.now(timezone.utc).isoformat(),
        "mode":           "multi-algorithm comparison",
        "algorithms":     all_algorithms,
        "t1_ab_present":  t1_present,
        "comparison":     comparison_tiers,
        "traceability": {
            "observer":       "2D Channelized Hotelling Observer (CHO)",
            "channels":       "10 Laguerre-Gauss (tier-specific a)",
            "covariance":     "LA condition only (§A1.5.2(a))",
            "regularization": "Tikhonov λ = 0.01 × trace(K) / 10",
            "hold_out":       "Leave-One-Out (LOO) with observer re-fit per fold",
            "bootstrap":      "1000 resamples, observation-level, on LOO scores",
            "wilcoxon":       "One-tailed paired Wilcoxon on d[i] = s_LP[i] − s_LA[i]",
            "estimation_bias":"50/50 2-fold CV",
            "metric":         "Mann-Whitney AUC (mid-rank tie correction)",
            "note_3d":        "3D integration across z shall not be performed (§A1.5.3)",
        },
    }


def generate_comparison_pdf(
    tdp: dict,
    output_path: Path,
) -> None:
    """Generate a multi-algorithm comparison TDP PDF report."""
    _register_fonts()

    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
        leftMargin=inch, rightMargin=inch,
    )
    styles = getSampleStyleSheet()

    T      = ParagraphStyle("T",   parent=styles["Title"],    fontName=PDF_FONT_BOLD, fontSize=16, spaceAfter=6)
    H1     = ParagraphStyle("H1",  parent=styles["Heading1"], fontName=PDF_FONT_BOLD, fontSize=13, spaceBefore=14, spaceAfter=4)
    H2     = ParagraphStyle("H2",  parent=styles["Heading2"], fontName=PDF_FONT_BOLD, fontSize=11, spaceBefore=8,  spaceAfter=3)
    B      = ParagraphStyle("B",   parent=styles["Normal"],   fontName=PDF_FONT,      fontSize=10, leading=14, spaceAfter=6)
    TC     = ParagraphStyle("TC",  parent=styles["Normal"],   fontName=PDF_FONT,      fontSize=9,  leading=11)
    TC_HDR = ParagraphStyle("TCH", parent=styles["Normal"],   fontName=PDF_FONT_BOLD, fontSize=9,  leading=11, textColor=colors.white)
    MONO   = ParagraphStyle("MN",  parent=styles["Normal"],   fontName="Courier",     fontSize=8,  leading=11, spaceAfter=6,
                             backColor=colors.HexColor("#F5F5F5"))

    def _cell(v, hdr=False, color=None):
        style = TC_HDR if hdr else TC
        if color:
            style = ParagraphStyle("colored", parent=TC, textColor=color)
        return Paragraph(str(v), style)

    def _tbl(data, widths):
        wrapped = [[_cell(c, hdr=(r == 0)) for c in row] for r, row in enumerate(data)]
        t = Table(wrapped, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#003366")),
            ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
            ("VALIGN",         (0,0), (-1,-1), "TOP"),
            ("GRID",           (0,0), (-1,-1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF2F7")]),
            ("TOPPADDING",     (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
            ("LEFTPADDING",    (0,0), (-1,-1), 5),
            ("RIGHTPADDING",   (0,0), (-1,-1), 5),
        ]))
        return t

    story = []

    # ── Cover ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Technical Dossier Package", T))
    story.append(Paragraph(
        f"Multi-Algorithm MAR Comparison — {STANDARD_REF}", styles["Heading2"],
    ))
    story.append(Paragraph(f"Generated {tdp['generation_date']}", styles["Italic"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey, spaceAfter=12))

    algos = tdp["algorithms"]
    story.append(Paragraph("1. Submission Identity", H1))
    story.append(_tbl([
        ["Field",              "Value"],
        ["Mode",               "Multi-algorithm comparison"],
        ["Algorithms",         ", ".join(algos)],
        ["Standard reference", STANDARD_REF],
        ["TDP version",        tdp["tdp_version"]],
        ["Generation date",    tdp["generation_date"]],
        ["T1_AB anchor tier",  "PRESENT" if tdp["t1_ab_present"] else "MISSING"],
    ], [2.5*inch, 4.1*inch]))
    story.append(Spacer(1, 8))

    # ── Per-tier comparison tables ─────────────────────────────────────────
    story.append(Paragraph("2. Algorithm Comparison by Tier", H1))

    for tier_data in tdp["comparison"]:
        tid = tier_data["tier_id"]
        tier = TIER_REGISTRY.get(tid)

        story.append(Paragraph(
            f"2.{['T1_AB','T2_SB','T3_HEAD'].index(tid)+1}  "
            f"Tier {tid} — {tier_data['description']}",
            H2,
        ))

        # Tier geometry summary
        if tier:
            body_desc = (
                f"circle r={tier.body_semi_x_mm:.0f} mm"
                if tier.is_circular_body
                else f"ellipse {tier.body_semi_x_mm:.0f}×{tier.body_semi_y_mm:.0f} mm"
            )
            story.append(Paragraph(
                f"{body_desc}  |  {tier.metal_material} μ={tier.metal_mu_cm:.2f} cm⁻¹  |  "
                f"blockage {tier.blockage_frac*100:.1f}%",
                B,
            ))

        # Comparison table header
        rows = [["Algorithm", "AUC_noMAR", "AUC_MAR", "ΔAUC", "95% CI ΔAUC", "Outcome"]]

        for algo in tier_data["algorithms"]:
            outcome = algo["outcome"]
            oc = _outcome_color(outcome)
            ci = algo["delta_AUC_CI"]
            rows.append([
                algo["algorithm"],
                f"{algo['AUC_noMAR']:.4f}" if algo["AUC_noMAR"] is not None else "—",
                f"{algo['AUC_MAR']:.4f}"   if algo["AUC_MAR"]   is not None else "—",
                f"{algo['delta_AUC']:+.4f}",
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
                outcome,
            ])

        story.append(_tbl(rows, [1.2*inch, 1.0*inch, 1.0*inch, 0.9*inch, 1.7*inch, 1.2*inch]))
        story.append(Spacer(1, 4))

        # Claim strings for each algorithm
        for algo in tier_data["algorithms"]:
            if algo["claim"]:
                oc = _outcome_color(algo["outcome"])
                story.append(Paragraph(
                    f"<font color='#{oc.hexval()[2:]}'><b>{algo['algorithm']} ({algo['outcome']}):</b></font>",
                    B,
                ))
                story.append(Paragraph(algo["claim"], MONO))
        story.append(Spacer(1, 8))

    # ── Methodology ───────────────────────────────────────────────────────
    story.append(Paragraph("3. Methodology Traceability", H1))
    tr = tdp["traceability"]
    rows = [["Parameter", "Value"]]
    for k, v in tr.items():
        rows.append([k.replace("_", " ").title(), str(v)])
    story.append(_tbl(rows, [2.2*inch, 4.4*inch]))

    doc.build(story)
    print(f"  PDF written → {output_path}")


# ─── PDF report ───────────────────────────────────────────────────────────────

def _register_fonts() -> None:
    if os.path.exists(_DEJAVU_REGULAR) and os.path.exists(_DEJAVU_BOLD):
        try:
            pdfmetrics.registerFont(TTFont(PDF_FONT,      _DEJAVU_REGULAR))
            pdfmetrics.registerFont(TTFont(PDF_FONT_BOLD, _DEJAVU_BOLD))
            return
        except Exception:
            pass
    globals()['PDF_FONT']      = "Helvetica"
    globals()['PDF_FONT_BOLD'] = "Helvetica-Bold"
    warnings.warn("DejaVu fonts not found — falling back to Helvetica.",
                  RuntimeWarning, stacklevel=3)


def _outcome_color(outcome: str) -> colors.Color:
    return {
        "SUPERIOR":     colors.HexColor("#006600"),
        "NON-INFERIOR": colors.HexColor("#005580"),
        "INDETERMINATE": colors.HexColor("#AA0000"),
    }.get(outcome, colors.black)


def generate_pdf_report(
    tdp: dict,
    output_path: Path,
    sweep_by_tier: dict[str, list[dict]] | None = None,
) -> None:
    """Generate the TDP PDF report."""
    _register_fonts()

    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
        leftMargin=inch, rightMargin=inch,
    )
    styles = getSampleStyleSheet()

    T      = ParagraphStyle("T",   parent=styles["Title"],    fontName=PDF_FONT_BOLD, fontSize=16, spaceAfter=6)
    H1     = ParagraphStyle("H1",  parent=styles["Heading1"], fontName=PDF_FONT_BOLD, fontSize=13, spaceBefore=14, spaceAfter=4)
    H2     = ParagraphStyle("H2",  parent=styles["Heading2"], fontName=PDF_FONT_BOLD, fontSize=11, spaceBefore=8,  spaceAfter=3)
    B      = ParagraphStyle("B",   parent=styles["Normal"],   fontName=PDF_FONT,      fontSize=10, leading=14, spaceAfter=6)
    TC     = ParagraphStyle("TC",  parent=styles["Normal"],   fontName=PDF_FONT,      fontSize=9,  leading=11)
    TC_HDR = ParagraphStyle("TCH", parent=styles["Normal"],   fontName=PDF_FONT_BOLD, fontSize=9,  leading=11, textColor=colors.white)
    MONO   = ParagraphStyle("MN",  parent=styles["Normal"],   fontName="Courier",     fontSize=8,  leading=11, spaceAfter=6,
                             backColor=colors.HexColor("#F5F5F5"))

    def _cell(v, hdr=False, color=None):
        style = TC_HDR if hdr else TC
        if color:
            style = ParagraphStyle("colored", parent=TC, textColor=color)
        return Paragraph(str(v), style)

    def _tbl(data, widths):
        wrapped = [[_cell(c, hdr=(r == 0)) for c in row] for r, row in enumerate(data)]
        t = Table(wrapped, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#003366")),
            ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
            ("VALIGN",         (0,0), (-1,-1), "TOP"),
            ("GRID",           (0,0), (-1,-1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF2F7")]),
            ("TOPPADDING",     (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
            ("LEFTPADDING",    (0,0), (-1,-1), 5),
            ("RIGHTPADDING",   (0,0), (-1,-1), 5),
        ]))
        return t

    mar_name = tdp["mar_implementation"]
    story = []

    # ── Cover ──────────────────────────────────────────────────────────────────
    story.append(Paragraph("Technical Dossier Package", T))
    story.append(Paragraph(
        f"Metal Artifact Reduction Performance Assessment — {STANDARD_REF}",
        styles["Heading2"],
    ))
    story.append(Paragraph(f"Generated {tdp['generation_date']}", styles["Italic"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey, spaceAfter=12))

    story.append(Paragraph("1. Submission Identity", H1))
    story.append(_tbl([
        ["Field",               "Value"],
        ["MAR implementation",  mar_name],
        ["Standard reference",  STANDARD_REF],
        ["TDP version",         tdp["tdp_version"]],
        ["Generation date",     tdp["generation_date"]],
        ["T1_AB anchor tier",   "PRESENT" if tdp["t1_ab_present"] else "MISSING (non-compliant)"],
    ], [2.5*inch, 4.1*inch]))
    story.append(Spacer(1, 8))

    if not tdp["t1_ab_present"]:
        story.append(Paragraph(
            "WARNING: T1_AB (large-body Co-Cr) is the mandatory anchor tier for "
            "510(k) submissions. This TDP does not include T1_AB results and "
            "is NOT compliant with the T1_AB requirement.",
            ParagraphStyle("warn", parent=B, backColor=colors.lightyellow,
                           borderPadding=(4, 6, 4, 6)),
        ))

    # ── Summary table ─────────────────────────────────────────────────────────
    story.append(Paragraph("2. Results Summary", H1))
    rows = [["Tier", "AUC_noMAR", "AUC_MAR", "ΔAUC", "95% CI ΔAUC", "Wilcoxon p", "Outcome"]]
    for t in tdp["tiers"]:
        outcome = t["outcome"]
        rows.append([
            t["tier_id"],
            f"{t['noMAR']['AUC_hold_out']:.4f}" if t["noMAR"]["AUC_hold_out"] is not None else "—",
            f"{t['MAR']['AUC_hold_out']:.4f}"   if t["MAR"]["AUC_hold_out"]   is not None else "—",
            f"{t['delta_AUC']:+.4f}",
            f"[{t['delta_AUC_CI_95'][0]:+.4f}, {t['delta_AUC_CI_95'][1]:+.4f}]",
            f"{t['MAR']['wilcoxon_p']:.4f}" if t["MAR"]["wilcoxon_p"] is not None else "—",
            outcome,
        ])
    story.append(_tbl(rows, [0.9*inch, 1.1*inch, 1.0*inch, 0.9*inch, 1.6*inch, 1.1*inch, 1.1*inch]))
    story.append(Spacer(1, 8))

    # ── Per-tier sections ──────────────────────────────────────────────────────
    story.append(Paragraph("3. Per-Tier Results", H1))

    for tier_res in tdp["tiers"]:
        tid = tier_res["tier_id"]
        tier = TIER_REGISTRY.get(tid)
        outcome = tier_res["outcome"]
        oc = _outcome_color(outcome)

        story.append(Paragraph(
            f"3.{list(TIER_REGISTRY.keys()).index(tid)+1 if tid in TIER_REGISTRY else '?'}  "
            f"Tier {tid} — {tier.description if tier else tid}",
            H2,
        ))

        # Tier geometry
        if tier:
            body_desc = (
                f"circle r={tier.body_semi_x_mm:.0f} mm"
                if tier.is_circular_body
                else f"ellipse {tier.body_semi_x_mm:.0f}×{tier.body_semi_y_mm:.0f} mm"
            )
            story.append(_tbl([
                ["Parameter",      "Value"],
                ["Body",           body_desc],
                ["Metal",          f"{tier.metal_material}  r={tier.metal_radius_mm:.0f} mm  μ={tier.metal_mu_cm:.2f} cm⁻¹"],
                ["Lesion",         f"{tier.lesion_semi_major_mm:.1f}×{tier.lesion_semi_minor_mm:.1f} mm  @ x={tier.lesion_center_x}"],
                ["Angular blockage", f"{tier.blockage_frac*100:.2f}%"],
                ["CHO ROI",        f"{tier.roi_size}×{tier.roi_size} vox  a={tier.channel_width_a:.1f} vox"],
                ["σ_internal",     str(tier_res.get("internal_noise_sigma", 15))],
            ], [2.2*inch, 4.4*inch]))
            story.append(Spacer(1, 4))

        # AUC table
        noMAR = tier_res["noMAR"]
        MAR   = tier_res["MAR"]
        auc_rows = [
            ["Condition", "AUC (LOO)", "Bias (resub−LOO)", "95% CI", "Wilcoxon p (1-tail)"],
        ]
        for lbl, d in (("noMAR", noMAR), ("MAR", MAR)):
            auc = d.get("AUC_hold_out")
            ci  = d.get("CI_95")
            auc_rows.append([
                lbl,
                f"{auc:.4f}" if auc is not None else "—",
                f"{d.get('bias_rs_minus_ho', 0):+.4f}",
                f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—",
                f"{d.get('wilcoxon_p', float('nan')):.4f}",
            ])
        story.append(_tbl(auc_rows, [0.8*inch, 1.1*inch, 1.5*inch, 1.8*inch, 1.4*inch]))
        story.append(Spacer(1, 4))

        # ΔAUC
        ci = tier_res["delta_AUC_CI_95"]
        story.append(Paragraph(
            f"ΔAUC = {tier_res['delta_AUC']:+.4f}  "
            f"(95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}];  "
            f"two-sided Wilcoxon p = {tier_res.get('delta_wilcoxon_p_twosided', float('nan')):.4f})",
            B,
        ))

        # Sigmoid fit summary
        sf = noMAR.get("sigmoid_fit")
        if sf and sf.get("fit_ok"):
            story.append(Paragraph(
                f"Sigmoid sensitivity (noMAR): σ₀ = {sf['sigma_0']:.1f}  "
                f"A = {sf['A']:.4f}  k = {sf['k']:.4f}  R² = {sf['r_squared']:.4f}",
                B,
            ))

        # Claim string
        claim_d = next((c for c in tdp["claims"] if c["tier_id"] == tid), None)
        if claim_d:
            story.append(Paragraph(
                f"<font color='#{oc.hexval()[2:]}'><b>Regulatory claim ({outcome}):</b></font>",
                B,
            ))
            story.append(Paragraph(claim_d["claim"], MONO))
        story.append(Spacer(1, 8))

    # ── Methodology ───────────────────────────────────────────────────────────
    story.append(Paragraph("4. Methodology Traceability", H1))
    tr = tdp["traceability"]
    rows = [["Parameter", "Value"]]
    for k, v in tr.items():
        rows.append([k.replace("_", " ").title(), str(v)])
    story.append(_tbl(rows, [2.2*inch, 4.4*inch]))

    # ── Vaishnav Sensitivity Matrix (optional) ────────────────────────────
    if sweep_by_tier:
        story.append(Spacer(1, 12))
        _build_sensitivity_section(story, sweep_by_tier, B, H1, H2, TC, TC_HDR)

    doc.build(story)
    print(f"  PDF written → {output_path}")


# ─── Vaishnav Sensitivity Matrix helpers ──────────────────────────────────────

def load_sweep_results(sweep_dir: Path) -> dict[str, list[dict]]:
    """
    Load all sweep JSON files from sweep_dir.

    Returns a dict keyed by tier_id mapping to a list of result dicts,
    each patched with 'contrast_factor' by sweep_vaishnav_matrix.py.
    """
    by_tier: dict[str, list[dict]] = {}
    for p in sorted(sweep_dir.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  Warning: could not load {p}: {exc}")
            continue
        tier_id = data.get("tier", {}).get("tier_id") or data.get("sweep_label", "")[:5]
        if not tier_id:
            continue
        by_tier.setdefault(tier_id, []).append(data)

    # Sort each tier's results by contrast_factor
    for tid in by_tier:
        by_tier[tid].sort(key=lambda d: float(d.get("contrast_factor", 1.0)))

    total = sum(len(v) for v in by_tier.values())
    print(f"  Loaded {total} sweep results across {len(by_tier)} tiers: "
          f"{', '.join(by_tier.keys())}")
    return by_tier


def _make_sensitivity_plot(sweep_items: list[dict], tier_id: str,
                           auc_threshold: float = 0.6) -> bytes:
    """
    Build the noMAR AUC(σ) sensitivity curves for one tier.

    Returns PNG bytes suitable for embedding in ReportLab via RLImage.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    n   = len(sweep_items)
    cmap = matplotlib.colormaps.get_cmap("plasma").resampled(max(n, 1))
    SIGMA_DENSE = np.linspace(0, 85, 300)

    for i, res in enumerate(sweep_items):
        cf  = res.get("contrast_factor", 1.0)
        ns  = res.get("noMAR", {}).get("noise_sweep", {})
        pts = ns.get("sweep_points", [])
        sf  = ns.get("sigmoid_fit", {})

        if not pts:
            continue

        sigmas = [p["sigma"] for p in pts]
        aucs   = [p["AUC"]   for p in pts]
        color  = cmap(i / max(n - 1, 1))

        ax.plot(sigmas, aucs, "o-", color=color, linewidth=1.6,
                markersize=4.5, label=f"CF={cf:.1f}", alpha=0.88, zorder=3)

        # Sigmoid fit overlay (dashed)
        if sf.get("fit_ok") and sf.get("A") is not None:
            A, k, s0 = sf["A"], sf["k"], sf["sigma_0"]
            y_fit = A / (1.0 + np.exp(k * (SIGMA_DENSE - s0))) + 0.5
            ax.plot(SIGMA_DENSE, y_fit, "--", color=color, linewidth=0.9,
                    alpha=0.55, zorder=2)
            # Inflection point marker
            ax.axvline(s0, color=color, alpha=0.12, linewidth=0.6, zorder=1)

    # Operational threshold
    ax.axhline(auc_threshold, color="black", linestyle=":",
               linewidth=1.1, alpha=0.6, zorder=4,
               label=f"AUC={auc_threshold:.1f} threshold")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.5,
               alpha=0.35, zorder=1)

    # Normative σ operating point
    ax.axvline(15, color="steelblue", linestyle="--", linewidth=1.0,
               alpha=0.45, zorder=1, label="σ=15 (normative)")

    ax.set_xlim(-2, 84)
    ax.set_ylim(0.48, 1.02)
    ax.set_xlabel("Internal noise σ (CHO observer)", fontsize=10)
    ax.set_ylabel("AUC_noMAR (LOO hold-out)", fontsize=10)
    ax.set_title(f"Vaishnav Sensitivity Matrix — {tier_id} (noMAR baseline)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.75,
              handlelength=1.8)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 40, 50, 65, 80])
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.22)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def _make_delta_auc_plot(sweep_items: list[dict], tier_id: str) -> bytes:
    """
    ΔAUC vs contrast factor at σ=15 (normative operating point).

    Shows how iMAR's relative performance changes with lesion contrast.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cfs   = [float(d.get("contrast_factor", 1.0)) for d in sweep_items]
    deltas = [float(d.get("delta_AUC", float("nan"))) for d in sweep_items]
    ci_lo  = [float(d["delta_AUC_CI_95"][0]) for d in sweep_items
              if "delta_AUC_CI_95" in d]
    ci_hi  = [float(d["delta_AUC_CI_95"][1]) for d in sweep_items
              if "delta_AUC_CI_95" in d]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    valid = [not (v != v) for v in deltas]   # NaN check
    cfs_v   = [c for c, v in zip(cfs,   valid) if v]
    deltas_v = [d for d, v in zip(deltas, valid) if v]

    ax.plot(cfs_v, deltas_v, "o-", color="steelblue",
            linewidth=1.8, markersize=5, label="ΔAUC (iMAR − noMAR)")

    if len(ci_lo) == len(cfs_v):
        ax.fill_between(cfs_v, ci_lo, ci_hi, alpha=0.15, color="steelblue",
                        label="95% CI")

    ax.axhline(0.0,   color="black",  linestyle="-",  linewidth=0.7, alpha=0.6)
    ax.axhline(0.05,  color="#006600", linestyle="--", linewidth=0.8, alpha=0.5,
               label="SUPERIOR threshold (CI_lo>0 → net +)")
    ax.axhline(-0.05, color="#AA0000", linestyle="--", linewidth=0.8, alpha=0.5,
               label="NON-INFERIOR floor")

    # Outcome region shading
    ylim_lo, ylim_hi = min(deltas_v + [-0.10]) - 0.02, max(deltas_v + [0.1]) + 0.02
    ax.set_ylim(ylim_lo, ylim_hi)
    ax.axhspan(0.0,   ylim_hi, alpha=0.04, color="green")
    ax.axhspan(-0.05, 0.0,     alpha=0.04, color="gold")
    ax.axhspan(ylim_lo, -0.05, alpha=0.04, color="red")

    ax.set_xlim(min(cfs_v) - 0.05, max(cfs_v) + 0.05)
    ax.set_xlabel("Contrast factor (CF)", fontsize=10)
    ax.set_ylabel("ΔAUC  (MAR − noMAR)  @ σ=15", fontsize=10)
    ax.set_title(f"iMAR Performance vs Lesion Contrast — {tier_id}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="best", framealpha=0.8)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.22)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def _build_sensitivity_section(
    story: list,
    sweep_by_tier: dict[str, list[dict]],
    B: Any,
    H1: Any,
    H2: Any,
    TC: Any,
    TC_HDR: Any,
) -> None:
    """
    Append Section 5 — Vaishnav Sensitivity Matrix to the PDF story.

    Parameters
    ----------
    story         : ReportLab story list to append to
    sweep_by_tier : output of load_sweep_results()
    B, H1, H2, TC, TC_HDR : ParagraphStyle objects from the calling context
    """
    def _cell(v, hdr=False, color=None):
        style = TC_HDR if hdr else TC
        if color:
            style = ParagraphStyle("colored", parent=TC, textColor=color)
        return Paragraph(str(v), style)

    def _tbl(data, widths):
        wrapped = [[_cell(c, hdr=(r == 0)) for c in row]
                   for r, row in enumerate(data)]
        t = Table(wrapped, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0),  colors.HexColor("#003366")),
            ("TEXTCOLOR",      (0,0), (-1,0),  colors.white),
            ("VALIGN",         (0,0), (-1,-1), "TOP"),
            ("GRID",           (0,0), (-1,-1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF2F7")]),
            ("TOPPADDING",     (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
            ("LEFTPADDING",    (0,0), (-1,-1), 5),
            ("RIGHTPADDING",   (0,0), (-1,-1), 5),
        ]))
        return t

    story.append(Paragraph("5. Vaishnav Sensitivity Matrix", H1))
    story.append(Paragraph(
        "AUC(σ) curves for each contrast factor (CF).  Dashed lines show the "
        "sigmoid fit AUC(σ) = A / (1 + exp(k(σ − σ₀))) + 0.5.  "
        "The normative operating point σ = 15 is marked.  "
        "The AUC = 0.60 line indicates the minimum useful detectability threshold.",
        B,
    ))

    for tid, items in sweep_by_tier.items():
        if not items:
            continue

        story.append(Paragraph(f"5.{list(sweep_by_tier.keys()).index(tid)+1}  "
                                f"Tier {tid}", H2))

        # ── Sensitivity curve plot ─────────────────────────────────────────
        try:
            png_sens = _make_sensitivity_plot(items, tid)
            story.append(RLImage(io.BytesIO(png_sens), width=6.8*inch, height=3.8*inch))
        except Exception as exc:
            story.append(Paragraph(f"[Plot unavailable: {exc}]", B))

        story.append(Spacer(1, 6))

        # ── ΔAUC vs contrast factor plot ───────────────────────────────────
        try:
            png_delta = _make_delta_auc_plot(items, tid)
            story.append(RLImage(io.BytesIO(png_delta), width=5.2*inch, height=3.1*inch))
        except Exception as exc:
            story.append(Paragraph(f"[ΔAUC plot unavailable: {exc}]", B))

        story.append(Spacer(1, 6))

        # ── Sigmoid fit R² table ───────────────────────────────────────────
        r2_rows = [["CF", "σ₀ (inflect.)", "A (amplitude)", "k (steepness)", "R²",
                    "ΔAUC @ σ=15", "Outcome"]]
        for res in items:
            cf  = res.get("contrast_factor", "?")
            ns  = res.get("noMAR", {}).get("noise_sweep", {})
            sf  = ns.get("sigmoid_fit", {})
            dA  = res.get("delta_AUC", float("nan"))
            ci  = res.get("delta_AUC_CI_95", [float("nan"), float("nan")])
            outcome = classify_outcome(dA, float(ci[0]), float(ci[1]))
            oc_color = _outcome_color(outcome)
            r2_rows.append([
                f"{cf:.1f}",
                f"{sf['sigma_0']:.1f}" if sf.get("fit_ok") else "—",
                f"{sf['A']:.4f}"       if sf.get("fit_ok") else "—",
                f"{sf['k']:.4f}"       if sf.get("fit_ok") else "—",
                f"{sf['r_squared']:.4f}" if sf.get("fit_ok") else "—",
                f"{dA:+.4f}",
                outcome,
            ])
        story.append(_tbl(r2_rows,
                          [0.55*inch, 1.1*inch, 1.1*inch, 1.1*inch, 0.75*inch,
                           1.0*inch, 1.05*inch]))
        story.append(Spacer(1, 10))

        # ── Operational envelope summary ───────────────────────────────────
        # Find where AUC at sigma=15 drops below 0.6 as CF increases
        auc_15 = []
        for res in items:
            cf = res.get("contrast_factor", 1.0)
            ns = res.get("noMAR", {}).get("noise_sweep", {})
            pts = {p["sigma"]: p["AUC"] for p in ns.get("sweep_points", [])}
            auc_15.append((cf, pts.get(15, float("nan"))))

        above_thresh = [(cf, a) for cf, a in auc_15 if a > 0.6]
        if above_thresh:
            lo_cf = min(cf for cf, _ in above_thresh)
            hi_cf = max(cf for cf, _ in above_thresh)
            story.append(Paragraph(
                f"Operational envelope (AUC_noMAR > 0.60 at σ=15): "
                f"CF ∈ [{lo_cf:.1f}, {hi_cf:.1f}].",
                B,
            ))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="ASTM WKXXXXX v1.0.0 — TDP Report Generator"
    )
    # ── Mode selection ─────────────────────────────────────────────────────
    ap.add_argument("--compare", action="store_true",
                    help="Multi-algorithm comparison mode: auto-discover all results "
                         "from --results-dir and generate a cross-algorithm comparison "
                         "report (default when --results is not specified)")
    # ── Single-algorithm mode ──────────────────────────────────────────────
    ap.add_argument("--results", nargs="+", default=None, metavar="JSON",
                    help="CHO results JSON file(s) from run_cho_analysis_v6_0.py "
                         "(one per tier; single-algorithm report)")
    ap.add_argument("--mar-name", default=None,
                    help="MAR algorithm name / version string (required for single-algo mode)")
    # ── Shared options ─────────────────────────────────────────────────────
    ap.add_argument("--results-dir", default=None, metavar="DIR",
                    help="Directory of result JSONs for auto-discovery "
                         f"(default: {_DEFAULT_RESULTS_DIR})")
    ap.add_argument("--output-dir", default="./tdp_output",
                    help="Directory for TDP JSON and PDF (default: ./tdp_output)")
    ap.add_argument("--sweep-dir", default=None, metavar="DIR",
                    help="Directory of sweep result JSONs from sweep_vaishnav_matrix.py; "
                         "adds Vaishnav Sensitivity Matrix section to the PDF")
    ap.add_argument("--no-pdf", action="store_true",
                    help="Skip PDF generation (write TDP JSON only)")
    args = ap.parse_args()

    # Default to --compare mode when --results is not given
    compare_mode = args.compare or (args.results is None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if compare_mode:
        # ── Multi-algorithm comparison mode ───────────────────────────────
        results_dir = Path(args.results_dir) if args.results_dir else _DEFAULT_RESULTS_DIR
        if not results_dir.exists():
            print(f"ERROR: results directory not found: {results_dir}")
            return

        print(f"Auto-discovering results from {results_dir} ...")
        by_tier = auto_discover_results(results_dir)

        if not by_tier:
            print("No valid results found. Exiting.")
            return

        # Build comparison TDP JSON
        tdp = build_comparison_tdp(by_tier)

        tdp_json_path = output_dir / "tdp_comparison.json"
        with open(tdp_json_path, "w") as f:
            json.dump(tdp, f, indent=2, default=str)
        print(f"\nTDP JSON → {tdp_json_path}")

        # Print comparison table to console
        print()
        print("=" * 72)
        print("Multi-Algorithm Comparison")
        print("=" * 72)
        for tier_data in tdp["comparison"]:
            tid = tier_data["tier_id"]
            print(f"\n{'─'*72}")
            print(f"  Tier {tid} — {tier_data['description']}")
            print(f"{'─'*72}")
            print(f"  {'Algorithm':<14s} {'AUC_noMAR':>10s} {'AUC_MAR':>10s} "
                  f"{'ΔAUC':>10s} {'95% CI ΔAUC':>22s}  Outcome")
            for algo in tier_data["algorithms"]:
                ci = algo["delta_AUC_CI"]
                auc_n = f"{algo['AUC_noMAR']:.4f}" if algo["AUC_noMAR"] is not None else "—"
                auc_m = f"{algo['AUC_MAR']:.4f}" if algo["AUC_MAR"] is not None else "—"
                print(f"  {algo['algorithm']:<14s} {auc_n:>10s} {auc_m:>10s} "
                      f"{algo['delta_AUC']:>+10.4f} "
                      f"[{ci[0]:+.4f}, {ci[1]:+.4f}]  {algo['outcome']}")
        print()
        print("=" * 72)

        # Generate comparison PDF
        if not args.no_pdf:
            pdf_path = output_dir / "TDP_Comparison_Report.pdf"
            generate_comparison_pdf(tdp, pdf_path)

    else:
        # ── Single-algorithm mode (original behavior) ─────────────────────
        if not args.mar_name:
            print("ERROR: --mar-name is required in single-algorithm mode")
            return

        results_list = []
        for path_str in args.results:
            p = Path(path_str)
            if not p.exists():
                print(f"ERROR: results file not found: {p}")
                continue
            with open(p) as f:
                results_list.append(json.load(f))
            print(f"Loaded: {p}")

        if not results_list:
            print("No valid results files found. Exiting.")
            return

        # Build TDP JSON
        tdp = build_tdp_json(results_list, args.mar_name)

        tdp_json_path = output_dir / "tdp.json"
        with open(tdp_json_path, "w") as f:
            json.dump(tdp, f, indent=2, default=str)
        print(f"TDP JSON → {tdp_json_path}")

        # Print claim strings to console
        print()
        print("=" * 72)
        print(f"TDP Claims — {args.mar_name}")
        print("=" * 72)
        for claim_d in tdp["claims"]:
            print(f"\n[{claim_d['tier_id']}] {claim_d['outcome']}")
            print(claim_d["claim"])
        print()
        if not tdp["t1_ab_present"]:
            print("WARNING: T1_AB mandatory anchor tier is missing from this TDP.")
        print("=" * 72)

        # Load sweep results (optional)
        sweep_by_tier = None
        if args.sweep_dir:
            sweep_path = Path(args.sweep_dir)
            if not sweep_path.exists():
                print(f"WARNING: --sweep-dir not found: {sweep_path} — skipping Section 5")
            else:
                print(f"\nLoading sweep results from {sweep_path} ...")
                sweep_by_tier = load_sweep_results(sweep_path)

        # Generate PDF
        if not args.no_pdf:
            pdf_path = output_dir / "TDP_Report.pdf"
            generate_pdf_report(tdp, pdf_path, sweep_by_tier=sweep_by_tier)

    print(f"\nTDP output → {output_dir}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
