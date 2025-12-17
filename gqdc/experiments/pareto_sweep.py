
# gqdc/experiments/pareto_sweep.py
"""
Pareto sweep over carbon-aware deferral aggressiveness to trade off Facility energy vs Wait/SLA.
- Starts from FIFO schedule, applies post-hoc deferrals (like emissions_aggressive v2),
  recomputes facility energy with MPC, and collects points (energy_kWh, wait_mean, sla_miss%).
- Saves: outputs/pareto_points.csv
- Plots: outputs/fig_pareto_energy_wait.png, outputs/fig_pareto_energy_sla.png
"""
import argparse, yaml, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gqdc.simulate.arrivals import generate_arrivals
from gqdc.simulate.carbon import carbon_signal
from gqdc.scheduler.baseline_fifo import Job, run_fifo
from gqdc.quantum.emulator import DEFAULT_WORKLOADS
from gqdc.experiments.simulate_facility import simulate_facility_timeline

def make_jobs(arrivals, workloads, deadline_s):
    rng = np.random.default_rng(0)
    jobs = []
    for i, a in enumerate(arrivals):
        w = rng.choice(workloads).name
        jobs.append(Job(i, a, w, deadline_s))
    return jobs

def mean_carbon_over_interval(carbon: dict, start: float, finish: float):
    s = int(np.floor(start)); f = int(np.ceil(finish))
    if f <= s:
        return float(carbon.get(s, 300.0))
    vals = [carbon.get(t, 300.0) for t in range(s, f)]
    return float(np.mean(vals)) if len(vals) else 300.0

def apply_deferrals(schedule_df: pd.DataFrame, carbon: dict,
                    deadline_s: float,
                    threshold: float = 480.0,
                    strict_threshold: bool = False,
                    forecast_min: int = 60,
                    drop_min: float = 40.0,
                    guard_min: float = 2.0,
                    deferral_step_min: int = 4,
                    max_deferrals: int = 8):
    df = schedule_df.copy().sort_values('arrival_min').reset_index(drop=True)
    if 'deadline_min' not in df.columns:
        df['deadline_min'] = df['arrival_min'] + (deadline_s/60.0)
    df['deferrals'] = 0
    df['wait_min'] = df.get('wait_min', 0.0)

    max_t = int(np.ceil(df['finish_min'].max()) + forecast_min + 5)
    c_arr = np.array([carbon.get(t, 300.0) for t in range(max_t)])

    for i,row in df.iterrows():
        start = row['start_min']; finish = row['finish_min']
        slack = df.at[i, 'deadline_min'] - finish
        d = 0
        cur_start = start; cur_finish = finish
        while d < max_deferrals:
            if slack <= guard_min: break
            now_c = mean_carbon_over_interval(carbon, cur_start, cur_finish)
            f_lo = int(cur_finish); f_hi = int(min(cur_finish+forecast_min, len(c_arr)-1))
            if f_hi <= f_lo: break
            future_c = float(np.mean(c_arr[f_lo:f_hi]))
            ok = (future_c <= now_c - drop_min) if not strict_threshold else ((now_c >= threshold) and (future_c < threshold) and (now_c - future_c >= drop_min))
            if not ok: break
            step = min(deferral_step_min, max(0.0, slack - guard_min))
            if step <= 0: break
            cur_start += step; cur_finish += step
            slack -= step; d += 1
        if d>0:
            df.at[i,'start_min'] = cur_start
            df.at[i,'finish_min'] = cur_finish
            df.at[i,'deferrals'] = d
            df.at[i,'wait_min'] = df.at[i,'start_min'] - df.at[i,'arrival_min']
    return df

def sla_miss_rate(df):
    if 'deadline_min' not in df.columns:
        return 0.0
    return float((df['finish_min'] > df['deadline_min']).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--deadline_s', type=float, default=None, help='override SLA deadline for sweep')
    # Carbon signal
    ap.add_argument('--carbon_base', type=float, default=650.0)
    ap.add_argument('--carbon_swing', type=float, default=400.0)
    # Deferral search grids
    ap.add_argument('--steps', type=str, default='2,3,4,5')
    ap.add_argument('--maxdefs', type=str, default='2,4,6,8')
    ap.add_argument('--drops', type=str, default='20,30,40,60')
    ap.add_argument('--forecasts', type=str, default='30,45,60,90')
    ap.add_argument('--strict', action='store_true')
    # Facility params (MPC used for energy)
    ap.add_argument('--fixed_setpoint', type=float, default=6.01)
    ap.add_argument('--cool_fraction', type=float, default=1.09)
    ap.add_argument('--cop_min_at6', type=float, default=2.32)
    ap.add_argument('--cop_max_at12', type=float, default=5.66)
    ap.add_argument('--ambient_base_c', type=float, default=21.1)
    ap.add_argument('--ambient_swing_c', type=float, default=7.11)
    ap.add_argument('--econ_max_gain', type=float, default=0.21)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    Path('outputs').mkdir(exist_ok=True)

    steps = [int(x) for x in args.steps.split(',') if x.strip()]
    maxdefs = [int(x) for x in args.maxdefs.split(',') if x.strip()]
    drops = [float(x) for x in args.drops.split(',') if x.strip()]
    forecasts = [int(x) for x in args.forecasts.split(',') if x.strip()]

    fac_params = dict(
        base_it_kw=1.5, heat_per_job_kw=0.4, cryo_kw=2.5,
        cool_fraction=args.cool_fraction, cop_min_at6=args.cop_min_at6,
        cop_max_at12=args.cop_max_at12, ambient_thresh_c=20.0,
        setpoint_thresh_c=10.0, econ_max_gain=args.econ_max_gain
    )

    rows = []
    for seed in range(args.seeds):
        # signals + baseline schedule
        arr = generate_arrivals(rate_per_min=cfg['signals']['arrivals']['rate_per_min'],
                                diurnal_amp=cfg['signals']['arrivals']['diurnal_amp'],
                                seed=seed)
        carbon = {i: c for i, c in enumerate(
            carbon_signal(base=args.carbon_base, swing=args.carbon_swing,
                          noise=cfg['signals']['carbon']['noise'], seed=seed))}
        d_s = args.deadline_s if args.deadline_s is not None else cfg['experiment']['sla_deadline_s']
        jobs = make_jobs(arr, DEFAULT_WORKLOADS, d_s)
        sch = run_fifo(jobs, carbon, cfg)
        base_df = pd.DataFrame(sch)
        # produce deadline column
        base_df['deadline_min'] = base_df['arrival_min'] + (d_s/60.0)

        # baseline energy with fixed setpoint (reference)
        from gqdc.experiments.simulate_facility import simulate_facility_timeline
        _, _, tot_fixed = simulate_facility_timeline(base_df, carbon, policy="fixed",
                                                     fixed_setpoint_c=args.fixed_setpoint,
                                                     facility_params=fac_params,
                                                     ambient_base_c=args.ambient_base_c,
                                                     ambient_swing_c=args.ambient_swing_c)

        for step in steps:
            for md in maxdefs:
                for dr in drops:
                    for fc in forecasts:
                        df_def = apply_deferrals(base_df, carbon, deadline_s=d_s,
                                                 threshold=480.0, strict_threshold=args.strict,
                                                 forecast_min=fc, drop_min=dr, guard_min=2.0,
                                                 deferral_step_min=step, max_deferrals=md)
                        # recompute energy with MPC on deferred timeline
                        ts, jd, tot_mpc = simulate_facility_timeline(df_def, carbon, policy="mpc",
                                                                     facility_params=fac_params,
                                                                     ambient_base_c=args.ambient_base_c,
                                                                     ambient_swing_c=args.ambient_swing_c)
                        rows.append(dict(
                            seed=seed, step=step, maxdef=md, drop=dr, forecast=fc, strict=args.strict,
                            energy_fixed_kwh=tot_fixed['facility_total_kwh'],
                            energy_mpc_kwh=tot_mpc['facility_total_kwh'],
                            energy_saving_pct=100.0*(tot_fixed['facility_total_kwh']-tot_mpc['facility_total_kwh'])/tot_fixed['facility_total_kwh'],
                            wait_mean=df_def['wait_min'].mean(),
                            sla_miss_rate=sla_miss_rate(df_def),
                            deferrals_per_job=df_def['deferrals'].mean()
                        ))
    out = pd.DataFrame(rows)
    out.to_csv('outputs/pareto_points.csv', index=False)

    # Aggregate across seeds
    g = out.groupby(['step','maxdef','drop','forecast','strict'], as_index=False).agg(
        energy_saving_pct=('energy_saving_pct','mean'),
        wait_mean=('wait_mean','mean'),
        sla_miss_rate=('sla_miss_rate','mean'),
        deferrals_per_job=('deferrals_per_job','mean')
    )

    # Plots
    plt.figure()
    plt.scatter(g['wait_mean'], g['energy_saving_pct'])
    plt.xlabel('Average wait (min)'); plt.ylabel('Energy saving (%)')
    plt.title('Pareto: Energy saving vs Wait')
    plt.grid(True); plt.tight_layout()
    plt.savefig('outputs/fig_pareto_energy_wait.png', dpi=180)

    plt.figure()
    plt.scatter(g['sla_miss_rate'], g['energy_saving_pct'])
    plt.xlabel('SLA miss rate'); plt.ylabel('Energy saving (%)')
    plt.title('Pareto: Energy saving vs SLA miss')
    plt.grid(True); plt.tight_layout()
    plt.savefig('outputs/fig_pareto_energy_sla.png', dpi=180)

    print("Saved: outputs/pareto_points.csv, fig_pareto_energy_wait.png, fig_pareto_energy_sla.png")

if __name__ == '__main__':
    main()
