#!/usr/bin/env python
"""
Generate conceptual figures for the GQDC paper:

- fig_system_overview.png
- fig_eval_workflow.png

Uses real energy numbers from facility_summary_ci.csv.
Works whether this script is placed in the project root OR in outputs/.
"""

import os
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


# ---------------------------------------------------------------------
# Paths & data helpers (robust to script location)
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try two possibilities:
# 1) script is in outputs/       -> CSV in SCRIPT_DIR
# 2) script is in project root   -> CSV in SCRIPT_DIR/outputs
candidate_out_dirs = [
    SCRIPT_DIR,
    os.path.join(SCRIPT_DIR, "outputs"),
]

OUT_DIR = None
FACILITY_CI_PATH = None

for d in candidate_out_dirs:
    csv_path = os.path.join(d, "facility_summary_ci.csv")
    if os.path.exists(csv_path):
        OUT_DIR = d
        FACILITY_CI_PATH = csv_path
        break

if OUT_DIR is None or FACILITY_CI_PATH is None:
    raise RuntimeError(
        "Could not find facility_summary_ci.csv.\n"
        "Tried:\n"
        f"  {os.path.join(SCRIPT_DIR, 'facility_summary_ci.csv')}\n"
        f"  {os.path.join(SCRIPT_DIR, 'outputs', 'facility_summary_ci.csv')}\n"
        "Make sure you have run the experiments and that facility_summary_ci.csv exists."
    )


def load_energy_summary():
    """
    Load facility energy CI summary and return:
    (fixed_kwh, mpc_kwh, reduction_pct, ci_lo_pct, ci_hi_pct, n)
    """
    df = pd.read_csv(FACILITY_CI_PATH)

    def get(metric_name, col):
        return df.loc[df["metric"] == metric_name, col].iloc[0]

    fixed_kwh = get("facility_total_kwh_fixed", "mean")
    mpc_kwh = get("facility_total_kwh_mpc", "mean")
    reduction_pct = get("reduction_pct", "mean")
    ci_lo_pct = get("reduction_pct", "ci95_lo")
    ci_hi_pct = get("reduction_pct", "ci95_hi")
    n = int(get("reduction_pct", "n"))

    return fixed_kwh, mpc_kwh, reduction_pct, ci_lo_pct, ci_hi_pct, n


# ---------------------------------------------------------------------
# Low-level drawing helpers
# ---------------------------------------------------------------------

def add_box(ax, xy, width, height, text, fontsize=9, facecolor="#f5f5f5"):
    """
    Draw a rectangle with centered text.
    xy = (x, y) is lower-left corner.
    """
    rect = Rectangle(
        xy,
        width,
        height,
        linewidth=1.0,
        edgecolor="black",
        facecolor=facecolor,
        zorder=1,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
        zorder=2,
    )
    return rect


def add_arrow(ax, xy_from, xy_to, text=None, fontsize=8):
    """
    Draw a straight arrow from xy_from (x, y) to xy_to (x, y).
    Optionally label it at the midpoint.
    """
    arrow = FancyArrowPatch(
        xy_from,
        xy_to,
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.0,
        color="black",
        zorder=0,
    )
    ax.add_patch(arrow)

    if text is not None:
        mx = (xy_from[0] + xy_to[0]) / 2.0
        my = (xy_from[1] + xy_to[1]) / 2.0
        ax.text(mx, my, text, fontsize=fontsize, ha="center", va="center")

    return arrow


# ---------------------------------------------------------------------
# Figure 1: System overview
# ---------------------------------------------------------------------

def make_fig_system_overview():
    """
    GQDC system overview: workload + signals -> scheduler + MPC -> facility -> metrics
    Annotated with real energy reduction numbers.
    """

    fixed_kwh, mpc_kwh, reduction, ci_lo, ci_hi, n = load_energy_summary()

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Layout parameters
    box_w = 3.0
    box_h = 1.5

    # Left column: workload + signals
    add_box(
        ax,
        xy=(0.5, 4.0),
        width=box_w,
        height=box_h,
        text="Workload arrivals\n(synthetic QPU jobs)",
    )

    add_box(
        ax,
        xy=(0.5, 1.5),
        width=box_w,
        height=box_h,
        text="Signals\n(carbon, price, ambient)",
    )

    # Middle: scheduler + MPC
    add_box(
        ax,
        xy=(5.0, 2.75),
        width=3.5,
        height=2.5,
        text="Carbon-aware scheduler\n(FIFO / EDF + deferrals)\n+\nMPC cooling setpoints",
    )

    # Facility
    add_box(
        ax,
        xy=(10.0, 2.75),
        width=3.0,
        height=2.5,
        text="Facility simulator\n(QPU + cryo + cooling)\nCOP: linear / nonlinear",
    )

    # Metrics & post-processing
    metrics_text = (
        "Metrics & post-processing\n"
        f"Fixed:  {fixed_kwh:.2f} kWh\n"
        f"MPC:    {mpc_kwh:.2f} kWh\n"
        f"ΔE: {reduction:.2f}%  "
        f"[{ci_lo:.2f}, {ci_hi:.2f}] (n={n})\n"
        "Emissions, fairness, cost"
    )
    add_box(
        ax,
        xy=(5.0, 0.25),
        width=8.0,
        height=2.0,
        text=metrics_text,
        fontsize=8,
        facecolor="#f0f8ff",
    )

    # Arrow from workload -> scheduler
    add_arrow(
        ax,
        xy_from=(0.5 + box_w, 4.0 + box_h / 2.0),
        xy_to=(5.0, 2.75 + box_h),
        text="jobs",
    )

    # Arrow from signals -> scheduler
    add_arrow(
        ax,
        xy_from=(0.5 + box_w, 1.5 + box_h / 2.0),
        xy_to=(5.0, 2.75),
        text="carbon, price,\nambient",
    )

    # Arrow scheduler -> facility
    add_arrow(
        ax,
        xy_from=(5.0 + 3.5, 2.75 + box_h),
        xy_to=(10.0, 2.75 + box_h),
        text="job start times\n+ setpoints",
    )

    # Arrow facility -> metrics
    add_arrow(
        ax,
        xy_from=(10.0 + 3.0 / 2.0, 2.75),
        xy_to=(9.0, 0.25 + 2.0),
        text="P(t), E, emissions,\nqueues",
    )

    ax.set_title("System overview: Green Quantum Data Center (GQDC)", fontsize=11)

    out_path = os.path.join(OUT_DIR, "fig_system_overview.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# Figure 2: Evaluation workflow
# ---------------------------------------------------------------------

def make_fig_eval_workflow():
    """
    Evaluation workflow figure: config -> simulations -> metrics -> plots + paper.
    Uses the same true energy reduction in an annotation, but is mostly schematic.
    """

    fixed_kwh, mpc_kwh, reduction, ci_lo, ci_hi, n = load_energy_summary()

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box_w = 3.2
    box_h = 1.4
    y_mid = 3.0

    # Boxes along a pipeline
    add_box(
        ax,
        xy=(0.5, y_mid),
        width=box_w,
        height=box_h,
        text="Configuration\n(YAML: workloads,\nMPC, deferrals)",
    )

    add_box(
        ax,
        xy=(4.0, y_mid),
        width=box_w,
        height=box_h,
        text="Per-seed simulations\n(8 seeds facility,\n5–6 seeds others)",
    )

    add_box(
        ax,
        xy=(7.5, y_mid),
        width=box_w,
        height=box_h,
        text="Metric extraction\nfacility_kWh, emissions,\nwait/SLA, fairness,\ncost",
    )

    add_box(
        ax,
        xy=(11.0, y_mid),
        width=box_w,
        height=box_h,
        text="Post-processing\nfig_*.png, tables,\nDOCX / LaTeX\n(paper.tex)",
    )

    # Arrows between pipeline steps
    add_arrow(
        ax,
        xy_from=(0.5 + box_w, y_mid + box_h / 2.0),
        xy_to=(4.0, y_mid + box_h / 2.0),
        text="run experiments.py",
    )

    add_arrow(
        ax,
        xy_from=(4.0 + box_w, y_mid + box_h / 2.0),
        xy_to=(7.5, y_mid + box_h / 2.0),
        text="write CSV/NPY\nin outputs/",
    )

    add_arrow(
        ax,
        xy_from=(7.5 + box_w, y_mid + box_h / 2.0),
        xy_to=(11.0, y_mid + box_h / 2.0),
        text="make_figures.py\n+ build_manuscript.py",
    )

    # A small annotation box using the real energy numbers
    summary_text = (
        "Example result (from facility_summary_ci.csv):\n"
        f"  Fixed   = {fixed_kwh:.2f} kWh\n"
        f"  MPC     = {mpc_kwh:.2f} kWh\n"
        f"  ΔE      = {reduction:.2f}% "
        f"[{ci_lo:.2f}, {ci_hi:.2f}] (n={n})"
    )
    add_box(
        ax,
        xy=(2.0, 0.7),
        width=9.0,
        height=1.6,
        text=summary_text,
        fontsize=8,
        facecolor="#eef7ff",
    )

    ax.set_title("Evaluation workflow and artifact generation", fontsize=11)

    out_path = os.path.join(OUT_DIR, "fig_eval_workflow.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    make_fig_system_overview()
    make_fig_eval_workflow()


if __name__ == "__main__":
    main()
