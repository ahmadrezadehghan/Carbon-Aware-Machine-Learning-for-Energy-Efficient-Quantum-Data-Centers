
# gqdc/experiments/plot_figures.py
# Requires matplotlib: pip install matplotlib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def emissions_bar():
    p = Path("outputs/compare_fifo_v2_summary.csv")
    if not p.exists():
        print("[plot] missing", p); return
    df = pd.read_csv(p)
    labels = df['label'].tolist()
    means = df['emissions_mean'].tolist()
    ci = df['emissions_ci95'].tolist()
    x = range(len(labels))
    plt.figure()
    plt.bar(x, means, yerr=ci, capsize=4)
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel("Emissions index (a.u.)")
    plt.title("Emissions by policy (mean ±95% CI)")
    Path("outputs").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("outputs/fig_emissions_bar.png", dpi=160)
    plt.close()
    print("[plot] saved outputs/fig_emissions_bar.png")

def facility_bar():
    p = Path("outputs/compare_facility_runs.csv")
    if not p.exists():
        print("[plot] missing", p); return
    df = pd.read_csv(p)
    g = df.groupby('policy')['facility_total_kwh'].agg(['mean','std','count']).reset_index()
    g['ci'] = 1.96 * g['std'] / (g['count'] ** 0.5)
    x = range(len(g))
    plt.figure()
    plt.bar(x, g['mean'], yerr=g['ci'], capsize=4)
    plt.xticks(x, g['policy'], rotation=0)
    plt.ylabel("Facility total energy (kWh)")
    plt.title("Facility energy by policy (mean ±95% CI across seeds)")
    Path("outputs").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("outputs/fig_facility_energy_bar.png", dpi=160)
    plt.close()
    print("[plot] saved outputs/fig_facility_energy_bar.png")

def facility_timeseries():
    p1 = Path("outputs/facility_ts_fixed.csv")
    p2 = Path("outputs/facility_ts_smart.csv")
    if not (p1.exists() and p2.exists()):
        print("[plot] missing timeseries CSVs"); return
    d1 = pd.read_csv(p1)
    d2 = pd.read_csv(p2)

    # Plot facility kW over time (one figure)
    plt.figure()
    plt.plot(d1['minute'], d1['facility_kw'], label="fixed")
    plt.plot(d2['minute'], d2['facility_kw'], label="smart")
    plt.xlabel("Minute")
    plt.ylabel("Facility kW")
    plt.title("Facility power over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/fig_facility_kw_timeseries.png", dpi=160)
    plt.close()
    print("[plot] saved outputs/fig_facility_kw_timeseries.png")

    # Plot setpoint over time (another figure)
    plt.figure()
    plt.plot(d1['minute'], d1['setpoint_c'], label="fixed")
    plt.plot(d2['minute'], d2['setpoint_c'], label="smart")
    plt.xlabel("Minute")
    plt.ylabel("Chilled water setpoint (°C)")
    plt.title("Setpoint over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/fig_setpoint_timeseries.png", dpi=160)
    plt.close()
    print("[plot] saved outputs/fig_setpoint_timeseries.png")

if __name__ == "__main__":
    emissions_bar()
    facility_bar()
    facility_timeseries()
