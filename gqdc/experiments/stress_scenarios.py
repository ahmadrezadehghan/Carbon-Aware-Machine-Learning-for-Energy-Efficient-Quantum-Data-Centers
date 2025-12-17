
# gqdc/experiments/stress_scenarios.py  (FIXED: safe config handling + custom high_load path)
"""
Run stress scenarios: high-load and heat wave, report facility energy reduction.
Outputs:
- outputs/stress_summary.csv
- outputs/fig_stress_bar.png
"""
import argparse, yaml
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gqdc.experiments.compare_facility_ci import run_once as run_once_cf, ci95
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

def run_custom_arrival(cfg, seed, arrival_rate_mult, ambient_base_c, ambient_swing_c,
                       fixed_setpoint, fac_params):
    # arrivals
    rate = cfg.get('signals',{}).get('arrivals',{}).get('rate_per_min', 0.9) * arrival_rate_mult
    diurnal = cfg.get('signals',{}).get('arrivals',{}).get('diurnal_amp', 0.15)
    arr = generate_arrivals(rate_per_min=rate, diurnal_amp=diurnal, seed=seed)
    # carbon
    c_cfg = cfg.get('signals',{}).get('carbon',{})
    c_base = c_cfg.get('base', 650.0); c_swing = c_cfg.get('swing', 400.0); c_noise = c_cfg.get('noise', 8.0)
    carbon = {i: c for i, c in enumerate(carbon_signal(base=c_base, swing=c_swing, noise=c_noise, seed=seed))}
    # schedule
    jobs = make_jobs(arr, cfg.get('experiment',{}).get('sla_deadline_s', 180.0))
    sch = run_fifo(jobs, carbon, cfg)
    df = pd.DataFrame(sch)
    # facility totals
    ts_f, jd_f, tot_f = simulate_facility_timeline(df, carbon, policy="fixed",
                                                   fixed_setpoint_c=fixed_setpoint,
                                                   facility_params=fac_params,
                                                   ambient_base_c=ambient_base_c,
                                                   ambient_swing_c=ambient_swing_c)
    ts_m, jd_m, tot_m = simulate_facility_timeline(df, carbon, policy="mpc",
                                                   facility_params=fac_params,
                                                   ambient_base_c=ambient_base_c,
                                                   ambient_swing_c=ambient_swing_c)
    return float(tot_f['facility_total_kwh']), float(tot_m['facility_total_kwh'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--seeds', type=int, default=6)
    # base facility params (same as CI script)
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

    scenarios = [
        ('default', dict(ambient_base_c=args.ambient_base_c, ambient_swing_c=args.ambient_swing_c, mode='run_once')),
        ('high_load', dict(arrival_rate_mult=1.8, ambient_base_c=args.ambient_base_c, ambient_swing_c=args.ambient_swing_c, mode='custom')),
        ('heat_wave', dict(ambient_base_c=args.ambient_base_c+8.0, ambient_swing_c=args.ambient_swing_c+3.0, mode='run_once')),
    ]

    rows = []
    for name, sc in scenarios:
        fixed_list, mpc_list = [], []
        for s in range(args.seeds):
            if sc['mode'] == 'run_once':
                f, m = run_once_cf(cfg, s, args.fixed_setpoint, fac_params,
                                   sc['ambient_base_c'], sc['ambient_swing_c'])
            else:
                f, m = run_custom_arrival(cfg, s, sc.get('arrival_rate_mult', 1.0),
                                          sc['ambient_base_c'], sc['ambient_swing_c'],
                                          args.fixed_setpoint, fac_params)
            fixed_list.append(f); mpc_list.append(m)
        fixed_arr = np.array(fixed_list); mpc_arr = np.array(mpc_list)
        red = 100.0*(fixed_arr - mpc_arr)/np.maximum(fixed_arr,1e-9)
        lo, hi = ci95(red)
        rows.append(dict(scenario=name, mean=float(np.mean(red)), ci95_lo=float(lo), ci95_hi=float(hi)))
    out = pd.DataFrame(rows)
    Path('outputs').mkdir(exist_ok=True)
    out.to_csv('outputs/stress_summary.csv', index=False)
    # plot
    plt.figure()
    plt.bar(out['scenario'], out['mean'])
    plt.ylabel('Energy reduction (%)'); plt.title('Stress scenarios')
    plt.tight_layout(); plt.savefig('outputs/fig_stress_bar.png', dpi=180)
    print("Saved: outputs/stress_summary.csv, fig_stress_bar.png")

if __name__ == '__main__':
    main()
