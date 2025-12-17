
# gqdc/experiments/cost_compare.py
"""
Compare cost (EUR) for fixed vs MPC using a simple day-night tariff.
Price signal: base 0.10 €/kWh + diurnal swing 0.05 €/kWh.
Outputs:
- outputs/cost_compare.csv
- outputs/fig_cost_bar.png
"""
import argparse, yaml, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gqdc.simulate.arrivals import generate_arrivals
from gqdc.simulate.carbon import carbon_signal
from gqdc.scheduler.baseline_fifo import Job, run_fifo
from gqdc.quantum.emulator import DEFAULT_WORKLOADS
from gqdc.experiments.simulate_facility import simulate_facility_timeline

def make_jobs(arrivals, deadline_s):
    import numpy as np
    rng = np.random.default_rng(0)
    jobs = []
    for i, a in enumerate(arrivals):
        w = rng.choice(DEFAULT_WORKLOADS).name
        jobs.append(Job(i, a, w, deadline_s))
    return jobs

def price_signal(n_min, base=0.10, swing=0.05, phase=0.0):
    import numpy as np
    t = np.arange(n_min)
    # diurnal: 1440 min period
    return base + swing * np.sin(2*np.pi*t/1440.0 + phase)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--seeds', type=int, default=5)
    # facility params
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
        sch = run_fifo(jobs, carbon, cfg)
        df = pd.DataFrame(sch)

        ts_f, jd_f, tot_f = simulate_facility_timeline(df, carbon, policy="fixed",
                                                       fixed_setpoint_c=args.fixed_setpoint,
                                                       facility_params=fac_params,
                                                       ambient_base_c=args.ambient_base_c,
                                                       ambient_swing_c=args.ambient_swing_c)
        ts_m, jd_m, tot_m = simulate_facility_timeline(df, carbon, policy="mpc",
                                                       facility_params=fac_params,
                                                       ambient_base_c=args.ambient_base_c,
                                                       ambient_swing_c=args.ambient_swing_c)
        nmin = int(max(ts_f['minute'].max(), ts_m['minute'].max())) + 1
        price = price_signal(nmin)
        # cost in EUR
        cost_f = float(np.sum(ts_f['facility_kw'].values/60.0 * price[:len(ts_f)]))
        cost_m = float(np.sum(ts_m['facility_kw'].values/60.0 * price[:len(ts_m)]))
        rows.append(dict(seed=seed,
                         energy_fixed_kwh=float(tot_f['facility_total_kwh']),
                         energy_mpc_kwh=float(tot_m['facility_total_kwh']),
                         cost_fixed_eur=cost_f, cost_mpc_eur=cost_m,
                         energy_saving_pct=100.0*(tot_f['facility_total_kwh']-tot_m['facility_total_kwh'])/tot_f['facility_total_kwh'],
                         cost_saving_pct=100.0*(cost_f - cost_m)/max(cost_f,1e-9)))
    res = pd.DataFrame(rows)
    Path('outputs').mkdir(exist_ok=True)
    res.to_csv('outputs/cost_compare.csv', index=False)

    # Plot bar means for cost
    plt.figure()
    plt.bar(['Fixed','MPC'], [res['cost_fixed_eur'].mean(), res['cost_mpc_eur'].mean()])
    plt.ylabel('Cost (EUR)'); plt.title('Cost comparison (day-night tariff)')
    plt.tight_layout(); plt.savefig('outputs/fig_cost_bar.png', dpi=180)
    print("Saved: outputs/cost_compare.csv, fig_cost_bar.png")

if __name__ == '__main__':
    main()
