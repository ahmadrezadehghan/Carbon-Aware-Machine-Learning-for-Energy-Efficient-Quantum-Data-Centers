
# gqdc/experiments/compare_facility_ci.py  (FIXED: pass carbon to simulate_facility_timeline)
import argparse, yaml, numpy as np, pandas as pd
from pathlib import Path
from gqdc.experiments.simulate_facility import make_jobs, simulate_facility_timeline
from gqdc.simulate.arrivals import generate_arrivals
from gqdc.simulate.carbon import carbon_signal
from gqdc.scheduler.baseline_fifo import run_fifo
from gqdc.quantum.emulator import DEFAULT_WORKLOADS

def run_once(cfg, seed, fixed_setpoint, fac_params, ambient_base_c, ambient_swing_c):
    arr = generate_arrivals(rate_per_min=cfg['signals']['arrivals']['rate_per_min'],
                            diurnal_amp=cfg['signals']['arrivals']['diurnal_amp'],
                            seed=seed)
    carbon = {i: c for i, c in enumerate(
        carbon_signal(base=cfg['signals']['carbon']['base'],
                      swing=cfg['signals']['carbon']['swing'],
                      noise=cfg['signals']['carbon']['noise'],
                      seed=seed))}
    jobs = make_jobs(arr, DEFAULT_WORKLOADS, cfg['experiment']['sla_deadline_s'])
    merged_cfg = cfg | {'carbon_threshold': cfg.get('experiment', {}).get('carbon_threshold', 1e9)}
    sch = run_fifo(jobs, carbon, merged_cfg)
    sch_df = pd.DataFrame(sch)

    # NOTE: your simulate_facility_timeline requires carbon as 2nd arg (Patch 6)
    ts_f, jd_f, tot_f = simulate_facility_timeline(sch_df, carbon,
                                                   policy="fixed",
                                                   fixed_setpoint_c=fixed_setpoint,
                                                   facility_params=fac_params,
                                                   ambient_base_c=ambient_base_c,
                                                   ambient_swing_c=ambient_swing_c)
    ts_m, jd_m, tot_m = simulate_facility_timeline(sch_df, carbon,
                                                   policy="mpc",
                                                   facility_params=fac_params,
                                                   ambient_base_c=ambient_base_c,
                                                   ambient_swing_c=ambient_swing_c)
    return float(tot_f['facility_total_kwh']), float(tot_m['facility_total_kwh'])

def ci95(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n < 2:
        return (np.nan, np.nan)
    m = arr.mean()
    s = arr.std(ddof=1)
    half = 1.96 * s / np.sqrt(n)
    return (m - half, m + half)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--seeds', type=int, default=8)
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

    fixed_list, mpc_list, red_list = [], [], []
    for s in range(args.seeds):
        f, m = run_once(cfg, s, args.fixed_setpoint, fac_params,
                        args.ambient_base_c, args.ambient_swing_c)
        fixed_list.append(f); mpc_list.append(m)
        red_list.append(100.0*(f-m)/max(f,1e-9))

    def _row(name, values):
        lo, hi = ci95(values)
        return dict(metric=name, mean=np.mean(values), ci95_lo=lo, ci95_hi=hi, n=len(values))

    rows = [
        _row('facility_total_kwh_fixed', fixed_list),
        _row('facility_total_kwh_mpc', mpc_list),
        _row('reduction_pct', red_list)
    ]
    out = pd.DataFrame(rows)
    Path('outputs').mkdir(exist_ok=True)
    out.to_csv('outputs/facility_summary_ci.csv', index=False)
    print(out)

if __name__ == '__main__':
    main()
