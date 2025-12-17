
# gqdc/experiments/slack_diagnostics.py
"""
Compute and print basic stats about SLA slack in the baseline FIFO schedule.
Helps you see why deferrals might be rare.
"""
import argparse, yaml, numpy as np, pandas as pd
from gqdc.simulate.arrivals import generate_arrivals
from gqdc.simulate.carbon import carbon_signal
from gqdc.scheduler.baseline_fifo import Job, run_fifo
from gqdc.quantum.emulator import DEFAULT_WORKLOADS

def make_jobs(arrivals, workloads, deadline_s):
    import numpy as np
    rng = np.random.default_rng(0)
    jobs = []
    for i, a in enumerate(arrivals):
        w = rng.choice(workloads).name
        jobs.append(Job(i, a, w, deadline_s))
    return jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    arr = generate_arrivals(rate_per_min=cfg['signals']['arrivals']['rate_per_min'],
                            diurnal_amp=cfg['signals']['arrivals']['diurnal_amp'])
    carbon = {i: c for i, c in enumerate(
        carbon_signal(base=cfg['signals']['carbon']['base'],
                      swing=cfg['signals']['carbon']['swing'],
                      noise=cfg['signals']['carbon']['noise']))}
    jobs = make_jobs(arr, DEFAULT_WORKLOADS, cfg['experiment']['sla_deadline_s'])
    sch = run_fifo(jobs, carbon, cfg)
    df = pd.DataFrame(sch)
    df['deadline_min'] = df['arrival_min'] + (cfg['experiment']['sla_deadline_s']/60.0)
    df['slack_min'] = df['deadline_min'] - df['finish_min']

    print(df[['job_id','arrival_min','start_min','finish_min','deadline_min','slack_min']].head(12))
    print("\nSlack stats (minutes):")
    print(df['slack_min'].describe())

if __name__ == '__main__':
    main()
