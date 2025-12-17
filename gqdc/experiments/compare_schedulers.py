
# gqdc/experiments/compare_schedulers.py
"""
Compare FIFO vs EDF (non-preemptive) using the same jobs and emulator stats.
We reconstruct EDF schedule from (arrival_min, runtime_s) and recompute emissions,
then simulate facility energy with MPC for both timelines.
Outputs:
- outputs/scheduler_compare.csv
- outputs/fig_scheduler_bar.png
"""
import argparse, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gqdc.simulate.arrivals import generate_arrivals
from gqdc.simulate.carbon import carbon_signal
from gqdc.scheduler.baseline_fifo import Job, run_fifo
from gqdc.quantum.emulator import DEFAULT_WORKLOADS
from gqdc.experiments.simulate_facility import simulate_facility_timeline

def make_jobs(arrivals, deadline_s):
    rng = np.random.default_rng(0)
    jobs = []
    for i, a in enumerate(arrivals):
        w = rng.choice(DEFAULT_WORKLOADS).name
        jobs.append(Job(i, a, w, deadline_s))
    return jobs

def mean_carbon_over_interval(carbon: dict, start: float, finish: float) -> float:
    s = int(np.floor(start)); f = int(np.ceil(finish))
    if f <= s:
        return float(carbon.get(s, 300.0))
    vals = [carbon.get(t, 300.0) for t in range(s, f)]
    return float(np.mean(vals)) if len(vals) else 300.0

def build_edf(df_fifo: pd.DataFrame, deadline_s: float) -> pd.DataFrame:
    # Use arrival_min and runtime_s (in seconds) from FIFO output to reconstruct EDF order
    req_cols = {'job_id','arrival_min','runtime_s'}
    if not req_cols.issubset(df_fifo.columns):
        raise RuntimeError(f"Missing columns in FIFO df: need {req_cols}")
    jobs = df_fifo[['job_id','arrival_min','runtime_s','energy_proxy']].copy()
    jobs['deadline_min'] = jobs['arrival_min'] + (deadline_s/60.0)
    jobs = jobs.sort_values('arrival_min').reset_index(drop=True)

    t = 0.0
    scheduled = []
    pending_idx = set(jobs.index.tolist())
    while pending_idx:
        # find available
        avail = [i for i in pending_idx if jobs.at[i,'arrival_min'] <= t + 1e-9]
        if not avail:
            # jump to next arrival
            nxt = min(pending_idx, key=lambda i: jobs.at[i,'arrival_min'])
            t = float(jobs.at[nxt,'arrival_min'])
            avail = [i for i in pending_idx if jobs.at[i,'arrival_min'] <= t + 1e-9]
        # pick earliest deadline
        pick = min(avail, key=lambda i: jobs.at[i,'deadline_min'])
        start = max(t, float(jobs.at[pick,'arrival_min']))
        run_min = float(jobs.at[pick,'runtime_s'])/60.0
        finish = start + run_min
        scheduled.append((int(jobs.at[pick,'job_id']), start, finish, float(jobs.at[pick,'arrival_min'])))
        t = finish
        pending_idx.remove(pick)

    out = pd.DataFrame(scheduled, columns=['job_id','start_min','finish_min','arrival_min'])
    out = out.merge(jobs[['job_id','energy_proxy','deadline_min']], on='job_id', how='left')
    out['wait_min'] = out['start_min'] - out['arrival_min']
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--seeds', type=int, default=5)
    # Facility params
    ap.add_argument('--fixed_setpoint', type=float, default=6.01)
    ap.add_argument('--cool_fraction', type=float, default=1.09)
    ap.add_argument('--cop_min_at6', type=float, default=2.32)
    ap.add_argument('--cop_max_at12', type=float, default=5.66)
    ap.add_argument('--ambient_base_c', type=float, default=21.1)
    ap.add_argument('--ambient_swing_c', type=float, default=7.11)
    ap.add_argument('--econ_max_gain', type=float, default=0.21)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))

    fac_params = dict(
        base_it_kw=1.5, heat_per_job_kw=0.4, cryo_kw=2.5,
        cool_fraction=args.cool_fraction, cop_min_at6=args.cop_min_at6,
        cop_max_at12=args.cop_max_at12, ambient_thresh_c=20.0,
        setpoint_thresh_c=10.0, econ_max_gain=args.econ_max_gain
    )

    rows = []
    for seed in range(args.seeds):
        arr = generate_arrivals(rate_per_min=cfg['signals']['arrivals']['rate_per_min'],
                                diurnal_amp=cfg['signals']['arrivals']['diurnal_amp'],
                                seed=seed)
        carbon = {i: c for i, c in enumerate(
            carbon_signal(base=cfg['signals']['carbon']['base'],
                          swing=cfg['signals']['carbon']['swing'],
                          noise=cfg['signals']['carbon']['noise'],
                          seed=seed))}
        jobs = make_jobs(arr, cfg['experiment']['sla_deadline_s'])
        fifo = run_fifo(jobs, carbon, cfg)
        df_fifo = pd.DataFrame(fifo)
        df_fifo['deadline_min'] = df_fifo['arrival_min'] + (cfg['experiment']['sla_deadline_s']/60.0)

        # Recompute emissions of FIFO using its energy_proxy
        df_fifo['emissions_index'] = df_fifo.apply(
            lambda r: r['energy_proxy'] * mean_carbon_over_interval(carbon, r['start_min'], r['finish_min']), axis=1)

        # EDF reconstruction
        df_edf = build_edf(df_fifo, cfg['experiment']['sla_deadline_s'])
        df_edf['emissions_index'] = df_edf.apply(
            lambda r: r['energy_proxy'] * mean_carbon_over_interval(carbon, r['start_min'], r['finish_min']), axis=1)

        # Facility simulations (fixed vs MPC)
        _, _, tot_fifo_mpc = simulate_facility_timeline(df_fifo, carbon, policy="mpc",
                                                        facility_params=fac_params,
                                                        ambient_base_c=args.ambient_base_c,
                                                        ambient_swing_c=args.ambient_swing_c)
        _, _, tot_edf_mpc = simulate_facility_timeline(df_edf, carbon, policy="mpc",
                                                       facility_params=fac_params,
                                                       ambient_base_c=args.ambient_base_c,
                                                       ambient_swing_c=args.ambient_swing_c)

        rows.append(dict(seed=seed,
                         fifo_energy_kwh=float(tot_fifo_mpc['facility_total_kwh']),
                         edf_energy_kwh=float(tot_edf_mpc['facility_total_kwh']),
                         fifo_wait=float(df_fifo['wait_min'].mean()),
                         edf_wait=float(df_edf['wait_min'].mean()),
                         fifo_sla=float((df_fifo['finish_min']>df_fifo['deadline_min']).mean()),
                         edf_sla=float((df_edf['finish_min']>df_edf['deadline_min']).mean())))

    res = pd.DataFrame(rows)
    Path('outputs').mkdir(exist_ok=True)
    res.to_csv('outputs/scheduler_compare.csv', index=False)

    # Plot bar means
    m_fifo = res['fifo_energy_kwh'].mean()
    m_edf = res['edf_energy_kwh'].mean()
    plt.figure()
    plt.bar(['FIFO (MPC)','EDF (MPC)'], [m_fifo, m_edf])
    plt.ylabel('Facility energy (kWh)'); plt.title('Scheduler comparison')
    plt.tight_layout(); plt.savefig('outputs/fig_scheduler_bar.png', dpi=180)
    print("Saved: outputs/scheduler_compare.csv, fig_scheduler_bar.png")

if __name__ == '__main__':
    main()
