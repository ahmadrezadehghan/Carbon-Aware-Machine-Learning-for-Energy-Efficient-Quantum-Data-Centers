
# gqdc/experiments/cop_nonlinear_ablation.py  (FIXED: robust numpy usage, no shadowing)
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
from gqdc.control.facility_energy_nl import facility_power_kw_nl as nl_power

def make_jobs(arrivals, workloads, deadline_s):
    rng = np.random.default_rng(0)
    jobs = []
    for i, a in enumerate(arrivals):
        w = rng.choice(workloads).name
        jobs.append(Job(i, a, w, deadline_s))
    return jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--seeds', type=int, default=5)
    # facility common params
    ap.add_argument('--fixed_setpoint', type=float, default=6.01)
    ap.add_argument('--cool_fraction', type=float, default=1.1)
    ap.add_argument('--cop_min_at6', type=float, default=2.4)
    ap.add_argument('--cop_max_at12', type=float, default=5.6)
    ap.add_argument('--ambient_base_c', type=float, default=22.0)
    ap.add_argument('--ambient_swing_c', type=float, default=6.5)
    ap.add_argument('--econ_max_gain', type=float, default=0.18)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
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
        jobs = make_jobs(arr, DEFAULT_WORKLOADS, cfg['experiment']['sla_deadline_s'])
        sch = run_fifo(jobs, carbon, cfg)
        df = pd.DataFrame(sch)

        fac_params = dict(
            base_it_kw=1.5, heat_per_job_kw=0.4, cryo_kw=2.5,
            cool_fraction=args.cool_fraction, cop_min_at6=args.cop_min_at6,
            cop_max_at12=args.cop_max_at12, ambient_thresh_c=20.0,
            setpoint_thresh_c=10.0, econ_max_gain=args.econ_max_gain
        )

        # linear: reuse simulate_facility_timeline with policy="mpc" (uses linear COP inside project)
        ts_l, jd_l, tot_l = simulate_facility_timeline(df, carbon, policy="mpc",
                                                       facility_params=fac_params,
                                                       ambient_base_c=args.ambient_base_c,
                                                       ambient_swing_c=args.ambient_swing_c)

        # nonlinear: rebuild timeline using nl_power but همان برنامه‌ریزی (MPC سبک داخلی)
        from gqdc.experiments.simulate_facility import build_running_map, ambient_signal
        horizon_min = int(np.ceil(df['finish_min'].max())) + 1
        running_map = build_running_map(df, horizon_min)
        amb = ambient_signal(horizon_min, base_c=args.ambient_base_c, swing_c=args.ambient_swing_c)

        def choose_sp(prev_sp, t, horizon=30, ramp=2.0):
            # کاندیداهای مجاز با محدودیت ramp
            cands = [min(max(prev_sp + d, 6.0), 12.0) for d in np.linspace(-ramp, ramp, 9)]
            if prev_sp not in cands: cands.append(prev_sp)
            best_sp, best_cost = prev_sp, float('inf')
            for sp in cands:
                cost = 0.0
                for k in range(horizon):
                    tt = t + k
                    if tt >= len(amb): break
                    rcount = len(running_map.get(tt, []))
                    kw = nl_power(rcount, sp, ambient_c=float(amb[tt]),
                                  base_it_kw=fac_params['base_it_kw'],
                                  heat_per_job_kw=fac_params['heat_per_job_kw'],
                                  cryo_kw=fac_params['cryo_kw'],
                                  cool_fraction=fac_params['cool_fraction'],
                                  econ_max_gain=fac_params['econ_max_gain'])
                    cost += kw
                if cost < best_cost:
                    best_cost, best_sp = cost, sp
            return float(best_sp)

        setpoint = 9.0
        ts_rows = []; job_energy = {int(j):0.0 for j in df['job_id'].tolist()}
        for t in range(horizon_min):
            rjobs = running_map[t]; rcount = len(rjobs)
            sp = choose_sp(setpoint, t)
            kw = nl_power(rcount, sp, ambient_c=float(amb[t]),
                          base_it_kw=fac_params['base_it_kw'],
                          heat_per_job_kw=fac_params['heat_per_job_kw'],
                          cryo_kw=fac_params['cryo_kw'],
                          cool_fraction=fac_params['cool_fraction'],
                          econ_max_gain=fac_params['econ_max_gain'])
            kwh = kw/60.0
            if rcount>0:
                share = kwh/rcount
                for jid in rjobs: job_energy[jid] += share
            ts_rows.append({'minute':t,'running_jobs':rcount,'setpoint_c':sp,'facility_kw':kw})
            setpoint = sp

        ts_nl = pd.DataFrame(ts_rows)
        tot_nl = dict(facility_total_kwh=ts_nl['facility_kw'].sum()/60.0,
                      facility_per_job_kwh_mean=float(np.mean(list(job_energy.values()))))

        rows.append(dict(seed=seed,
                         linear_kwh=float(tot_l['facility_total_kwh']),
                         nonlinear_kwh=float(tot_nl['facility_total_kwh']),
                         delta_pct=100.0*(float(tot_l['facility_total_kwh'])-float(tot_nl['facility_total_kwh']))/max(float(tot_l['facility_total_kwh']),1e-9)))

    out = pd.DataFrame(rows)
    Path('outputs').mkdir(exist_ok=True)
    out.to_csv('outputs/cop_ablation.csv', index=False)
    print(out.describe())

    # Plot bar of means
    m_lin = out['linear_kwh'].mean(); m_nl = out['nonlinear_kwh'].mean()
    plt.figure()
    plt.bar(['Linear COP','Nonlinear COP'], [m_lin, m_nl])
    plt.ylabel('Facility energy (kWh)'); plt.title('Linear vs Nonlinear COP')
    plt.tight_layout(); plt.savefig('outputs/fig_cop_ablation.png', dpi=180)
    print("Saved: outputs/cop_ablation.csv, fig_cop_ablation.png")

if __name__ == '__main__':
    main()
