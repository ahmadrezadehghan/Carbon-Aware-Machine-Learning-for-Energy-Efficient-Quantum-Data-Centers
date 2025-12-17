
import argparse, yaml, numpy as np, pandas as pd
from gqdc.simulate.arrivals import generate_arrivals
from gqdc.simulate.carbon import carbon_signal
from gqdc.scheduler.baseline_fifo import Job, run_fifo
from gqdc.quantum.emulator import DEFAULT_WORKLOADS

def make_jobs(arrivals, workloads, deadline_s):
    rng = np.random.default_rng(0)
    jobs = []
    for i, a in enumerate(arrivals):
        w = rng.choice(workloads).name
        jobs.append(Job(i, a, w, deadline_s))
    return jobs

def ci95(series: pd.Series):
    arr = series.dropna().to_numpy()
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    m = arr.mean()
    s = arr.std(ddof=1)
    n = len(arr)
    half = 1.96 * s / (n ** 0.5)
    return (m - half, m + half)

def main(path_cfg):
    cfg = yaml.safe_load(open(path_cfg))
    arr = generate_arrivals(rate_per_min=cfg['signals']['arrivals']['rate_per_min'],
                            diurnal_amp=cfg['signals']['arrivals']['diurnal_amp'])
    carbon = {i: c for i, c in enumerate(
        carbon_signal(base=cfg['signals']['carbon']['base'],
                      swing=cfg['signals']['carbon']['swing'],
                      noise=cfg['signals']['carbon']['noise']))}
    jobs = make_jobs(arr, DEFAULT_WORKLOADS, cfg['experiment']['sla_deadline_s'])
    merged_cfg = cfg | {'carbon_threshold': cfg.get('experiment', {}).get('carbon_threshold', 1e9)}
    res = run_fifo(jobs, carbon, merged_cfg)
    df = pd.DataFrame(res)

    e_ci = ci95(df['energy_proxy'])
    w_ci = ci95(df['wait_min'])
    em_ci = ci95(df['emissions_index'])
    sla_rate = df['sla_miss'].mean()

    print(df[['energy_proxy','emissions_index','wait_min','runtime_s','sla_miss']].describe())
    print(f"SLA miss rate: {sla_rate:.4f}")
    print(f"Energy_proxy mean ±95%CI: {df['energy_proxy'].mean():.2f} [{e_ci[0]:.2f}, {e_ci[1]:.2f}]")
    print(f"Emissions_index mean ±95%CI: {df['emissions_index'].mean():.2f} [{em_ci[0]:.2f}, {em_ci[1]:.2f}]")
    print(f"Wait_min mean ±95%CI: {df['wait_min'].mean():.3f} [{w_ci[0]:.3f}, {w_ci[1]:.3f}]")

    import os
    os.makedirs('outputs', exist_ok=True)
    df.to_csv('outputs/fifo_results.csv', index=False)
    print("Saved: outputs/fifo_results.csv")

if __name__ == '__main__':
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    args = ap.parse_args()
    main(args.config)
