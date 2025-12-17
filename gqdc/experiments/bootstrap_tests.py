
# gqdc/experiments/bootstrap_tests.py
"""
Bootstrap significance testing for facility energy reduction (MPC vs Fixed).
Re-runs seeds and computes bootstrap CI and p-value for mean reduction.
Outputs: outputs/bootstrap_energy.json and fig_bootstrap_hist.png
"""
import argparse, yaml
import numpy as np, pandas as pd, json
import matplotlib.pyplot as plt
from pathlib import Path
from gqdc.experiments.compare_facility_ci import ci95
from gqdc.experiments.compare_facility_ci import run_once as run_once_cf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--boots', type=int, default=5000)
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
    fixed_list, mpc_list, red_list = [], [], []
    for s in range(args.seeds):
        f, m = run_once_cf(cfg, s, args.fixed_setpoint, fac_params,
                           args.ambient_base_c, args.ambient_swing_c)
        fixed_list.append(f); mpc_list.append(m)
        red_list.append(100.0*(f-m)/max(f,1e-9))
    fixed_arr = np.array(fixed_list); mpc_arr = np.array(mpc_list)
    delta = fixed_arr - mpc_arr
    mean_red = 100.0*np.mean(delta/np.maximum(fixed_arr,1e-9))
    ci_lo, ci_hi = ci95(100.0*delta/np.maximum(fixed_arr,1e-9))

    # bootstrap
    rng = np.random.default_rng(0)
    vals = 100.0*(delta/np.maximum(fixed_arr,1e-9))
    idx = np.arange(len(vals))
    boots = []
    for _ in range(args.boots):
        b = rng.choice(idx, size=len(idx), replace=True)
        boots.append(np.mean(vals[b]))
    boots = np.array(boots)
    p_value = float(np.mean(boots <= 0.0))  # one-sided: mean reduction > 0

    Path('outputs').mkdir(exist_ok=True)
    with open('outputs/bootstrap_energy.json','w',encoding='utf-8') as f:
        json.dump(dict(mean_reduction_pct=float(mean_red),
                       ci95_lo=float(ci_lo), ci95_hi=float(ci_hi),
                       p_value=p_value), f, ensure_ascii=False, indent=2)

    plt.figure()
    plt.hist(boots, bins=40)
    plt.axvline(0, linestyle='--')
    plt.xlabel('Bootstrap mean reduction (%)'); plt.ylabel('count')
    plt.title(f'Bootstrap (p ≈ {p_value:.4f})')
    plt.tight_layout(); plt.savefig('outputs/fig_bootstrap_hist.png', dpi=180)
    print("Saved: outputs/bootstrap_energy.json, fig_bootstrap_hist.png")

if __name__ == '__main__':
    main()
