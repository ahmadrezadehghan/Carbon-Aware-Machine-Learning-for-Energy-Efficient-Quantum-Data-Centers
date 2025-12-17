
# gqdc/experiments/quick_stats.py  (UPDATED: handle 'mpc' or 'smart' labels)
import pandas as pd
from pathlib import Path

def summarize_emissions():
    p = Path("outputs/compare_fifo_v2_summary.csv")
    if not p.exists():
        print("[quick_stats] skip emissions (file not found):", p)
        return
    df = pd.read_csv(p)
    df = df[['label','n','emissions_mean','emissions_ci95','sla_rate','wait_mean']]
    print("\n=== Emissions (carbon-aware vs blind) ===")
    print(df.to_string(index=False))
    try:
        base = df.loc[df['label'].str.contains('A_blind'),'emissions_mean'].mean()
        smart = df.loc[df['label'].str.contains('B_'),'emissions_mean'].mean()
        if pd.notna(base) and pd.notna(smart):
            red = 100.0*(base-smart)/base
            print(f"Emissions reduction (smart vs blind): {red:.2f}%")
    except Exception as e:
        print("calc error:", e)

def summarize_facility():
    p = Path("outputs/compare_facility_runs.csv")
    if not p.exists():
        print("[quick_stats] skip facility energy (file not found):", p)
        return
    df = pd.read_csv(p)
    print("\n=== Facility energy (kWh) ===")
    g = df.groupby('policy')[['facility_total_kwh','facility_per_job_kwh_mean']].mean()
    print(g)
    try:
        fixed = df[df['policy']=='fixed']['facility_total_kwh'].mean()
        # accept either 'mpc' or 'smart' as the optimized policy
        if 'mpc' in set(df['policy']):
            opt = df[df['policy']=='mpc']['facility_total_kwh'].mean()
            label = 'mpc'
        else:
            opt = df[df['policy']=='smart']['facility_total_kwh'].mean()
            label = 'smart'
        red = 100.0*(fixed-opt)/fixed
        print(f"Average facility energy reduction ({label} vs fixed): {red:.2f}%")
    except Exception as e:
        print("calc error:", e)

if __name__ == "__main__":
    summarize_emissions()
    summarize_facility()
