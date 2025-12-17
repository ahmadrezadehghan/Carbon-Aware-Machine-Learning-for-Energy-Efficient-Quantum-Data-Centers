
# gqdc/experiments/make_tables.py
# Produces Markdown tables for the paper
import pandas as pd
from pathlib import Path

def table_emissions_md():
    p = Path("outputs/compare_fifo_v2_summary.csv")
    if not p.exists(): return ""
    df = pd.read_csv(p)[['label','n','emissions_mean','emissions_ci95','wait_mean','sla_rate']]
    df['emissions_ci'] = df['emissions_mean'].round(2).astype(str) + " ± " + df['emissions_ci95'].round(2).astype(str)
    df['wait_mean'] = df['wait_mean'].round(3)
    df['sla_rate'] = df['sla_rate'].round(3)
    out = df[['label','n','emissions_ci','wait_mean','sla_rate']]
    return out.to_markdown(index=False)

def table_facility_md():
    p = Path("outputs/compare_facility_runs.csv")
    if not p.exists(): return ""
    df = pd.read_csv(p)
    g = df.groupby('policy')[['facility_total_kwh','facility_per_job_kwh_mean']].mean().reset_index()
    g['facility_total_kwh'] = g['facility_total_kwh'].round(3)
    g['facility_per_job_kwh_mean'] = g['facility_per_job_kwh_mean'].round(4)
    return g.to_markdown(index=False)

if __name__ == "__main__":
    em = table_emissions_md()
    fa = table_facility_md()
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/paper_tables.md", "w", encoding="utf-8") as f:
        f.write("# Tables for paper\n\n")
        if em:
            f.write("## Emissions comparison\n\n" + em + "\n\n")
        if fa:
            f.write("## Facility energy comparison\n\n" + fa + "\n")
    print("[tables] written outputs/paper_tables.md")
