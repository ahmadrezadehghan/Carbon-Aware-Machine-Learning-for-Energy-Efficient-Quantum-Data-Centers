
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

def run_condition(cfg, label, arrival_rate=None, deadline_s=None, carbon_threshold=None,
                  deferral_step_min=None, forecast_window_min=None, drop_min=None,
                  guard_min=None, max_deferrals=None, seed=0):
    import copy
    cfg2 = copy.deepcopy(cfg)
    if arrival_rate is not None: cfg2['signals']['arrivals']['rate_per_min'] = float(arrival_rate)
    if deadline_s is not None: cfg2['experiment']['sla_deadline_s'] = int(deadline_s)
    if carbon_threshold is not None: cfg2['experiment']['carbon_threshold'] = float(carbon_threshold)
    # deferral knobs in top-level for simplicity
    if deferral_step_min is not None: cfg2['deferral_step_min'] = int(deferral_step_min)
    if forecast_window_min is not None: cfg2['forecast_window_min'] = int(forecast_window_min)
    if drop_min is not None: cfg2['drop_min'] = float(drop_min)
    if guard_min is not None: cfg2['guard_min'] = float(guard_min)
    if max_deferrals is not None: cfg2['max_deferrals'] = int(max_deferrals)

    arr = generate_arrivals(rate_per_min=cfg2['signals']['arrivals']['rate_per_min'],
                            diurnal_amp=cfg2['signals']['arrivals']['diurnal_amp'],
                            seed=seed)
    carbon = {i: c for i, c in enumerate(
        carbon_signal(base=cfg2['signals']['carbon']['base'],
                      swing=cfg2['signals']['carbon']['swing'],
                      noise=cfg2['signals']['carbon']['noise'],
                      seed=seed))}
    jobs = make_jobs(arr, DEFAULT_WORKLOADS, cfg2['experiment']['sla_deadline_s'])
    merged_cfg = cfg2 | {'carbon_threshold': cfg2['experiment'].get('carbon_threshold', 1e9)}
    res = run_fifo(jobs, carbon, merged_cfg)
    df = pd.DataFrame(res)
    df['label'] = label
    df['seed'] = seed
    return df

def summarize(df):
    g = df.groupby('label').agg(
        n=('job_id','count'),
        energy_mean=('energy_proxy','mean'),
        energy_std=('energy_proxy','std'),
        emissions_mean=('emissions_index','mean'),
        emissions_std=('emissions_index','std'),
        wait_mean=('wait_min','mean'),
        wait_std=('wait_min','std'),
        sla_rate=('sla_miss','mean')
    ).reset_index()
    g['energy_ci95'] = 1.96 * g['energy_std'] / np.sqrt(g['n'])
    g['emissions_ci95'] = 1.96 * g['emissions_std'] / np.sqrt(g['n'])
    g['wait_ci95']   = 1.96 * g['wait_std']   / np.sqrt(g['n'])
    return g

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))

    seeds = [0,1,2,3,4]
    frames = []
    for s in seeds:
        frames.append(run_condition(cfg, "A_blind", carbon_threshold=1e9, seed=s))
        # deadline-aware carbon deferral with sane knobs
        frames.append(run_condition(cfg, "B_carbon500_smart",
                                    carbon_threshold=500, deferral_step_min=2,
                                    forecast_window_min=15, drop_min=60, guard_min=2, max_deferrals=3,
                                    seed=s))
        frames.append(run_condition(cfg, "C_stress_smart",
                                    arrival_rate=0.7, deadline_s=60,
                                    carbon_threshold=500, deferral_step_min=2,
                                    forecast_window_min=15, drop_min=60, guard_min=2, max_deferrals=3,
                                    seed=s))
    df = pd.concat(frames, ignore_index=True)

    import os
    os.makedirs('outputs', exist_ok=True)
    df.to_csv('outputs/compare_fifo_v2.csv', index=False)

    summary = summarize(df)
    summary.to_csv('outputs/compare_fifo_v2_summary.csv', index=False)
    print(summary)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    args = ap.parse_args()
    main(args.config)
