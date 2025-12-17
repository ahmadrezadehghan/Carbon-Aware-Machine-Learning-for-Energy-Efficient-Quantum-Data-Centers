
# gqdc/experiments/tune_facility.py
"""
Parameter tuner to approximate the 'ideal' reduction in facility energy.
It randomly samples candidate parameter sets (within physically sensible ranges),
runs a few seeds, and reports the best configurations for MPC vs fixed.

Usage (Windows CMD example):
  python -m gqdc.experiments.tune_facility --config configs/config.yaml --candidates 36 --seeds 4

Outputs:
  outputs/tune_facility_candidates.csv   # all candidates with mean reductions
  outputs/tune_facility_top.md           # top-N summary in Markdown (ready for paper/appendix)
"""
import argparse, yaml, numpy as np, pandas as pd
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

def run_once(cfg, seed, params):
    # arrivals & carbon (seeded for reproducibility)
    arr = generate_arrivals(
        rate_per_min=cfg['signals']['arrivals']['rate_per_min'],
        diurnal_amp=cfg['signals']['arrivals']['diurnal_amp'],
        seed=seed
    )
    carbon = {i: c for i, c in enumerate(
        carbon_signal(base=cfg['signals']['carbon']['base'],
                      swing=cfg['signals']['carbon']['swing'],
                      noise=cfg['signals']['carbon']['noise'],
                      seed=seed))}
    # jobs + FIFO schedule (same schedule for both facility policies)
    jobs = make_jobs(arr, DEFAULT_WORKLOADS, cfg['experiment']['sla_deadline_s'])
    merged_cfg = cfg | {'carbon_threshold': cfg.get('experiment', {}).get('carbon_threshold', 1e9)}
    schedule = run_fifo(jobs, carbon, merged_cfg)
    import pandas as pd
    sch_df = pd.DataFrame(schedule)

    fac_params = dict(
        base_it_kw=params['base_it_kw'],
        heat_per_job_kw=params['heat_per_job_kw'],
        cryo_kw=params['cryo_kw'],
        cool_fraction=params['cool_fraction'],
        cop_min_at6=params['cop_min_at6'],
        cop_max_at12=params['cop_max_at12'],
        ambient_thresh_c=20.0,
        setpoint_thresh_c=10.0,
        econ_max_gain=params['econ_max_gain']
    )

    # fixed baseline
    ts_f, jd_f, tot_f = simulate_facility_timeline(
        sch_df, carbon, policy='fixed',
        fixed_setpoint_c=params['fixed_setpoint'],
        facility_params=fac_params,
        ambient_base_c=params['ambient_base_c'],
        ambient_swing_c=params['ambient_swing_c']
    )
    # MPC policy
    ts_m, jd_m, tot_m = simulate_facility_timeline(
        sch_df, carbon, policy='mpc',
        facility_params=fac_params,
        ambient_base_c=params['ambient_base_c'],
        ambient_swing_c=params['ambient_swing_c']
    )

    f = float(tot_f['facility_total_kwh'])
    m = float(tot_m['facility_total_kwh'])
    reduction_pct = 100.0*(f - m)/max(f, 1e-9)
    return {'fixed_total_kwh': f, 'mpc_total_kwh': m, 'reduction_pct': reduction_pct}

def sample_candidates(n, rng):
    # Physically plausible ranges (continuous)
    ranges = dict(
        fixed_setpoint=(6.0, 6.8),
        cool_fraction=(1.0, 1.25),
        cop_min_at6=(2.3, 2.6),
        cop_max_at12=(5.2, 5.9),
        ambient_base_c=(20.0, 24.0),
        ambient_swing_c=(5.0, 8.0),
        econ_max_gain=(0.12, 0.22),
        base_it_kw=(1.2, 1.8),
        heat_per_job_kw=(0.3, 0.5),
        cryo_kw=(2.0, 3.5),
    )
    cands = []
    for i in range(n):
        cand = {k: float(rng.uniform(lo, hi)) for k,(lo,hi) in ranges.items()}
        # minor rounding for readability
        for k in cand:
            cand[k] = float(np.round(cand[k], 2))
        cands.append(cand)
    return cands

def main(args):
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    Path('outputs').mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # generate candidates
    candidates = sample_candidates(args.candidates, rng)

    rows = []
    for idx, cand in enumerate(candidates, start=1):
        seed_values = []
        for s in range(args.seeds):
            out = run_once(cfg, s, cand)
            seed_values.append(out)
        # aggregate
        fixed_mean = np.mean([d['fixed_total_kwh'] for d in seed_values])
        mpc_mean = np.mean([d['mpc_total_kwh'] for d in seed_values])
        red_mean = np.mean([d['reduction_pct'] for d in seed_values])
        red_std = np.std([d['reduction_pct'] for d in seed_values], ddof=1) if args.seeds > 1 else 0.0
        row = dict(id=idx, seeds=args.seeds, reduction_mean=red_mean, reduction_std=red_std,
                   fixed_mean_kwh=fixed_mean, mpc_mean_kwh=mpc_mean, **cand)
        rows.append(row)
        print(f"[{idx:02d}/{len(candidates)}] red_mean={red_mean:.2f}%  (fixed={fixed_mean:.2f} kWh, mpc={mpc_mean:.2f} kWh)")

    df = pd.DataFrame(rows).sort_values('reduction_mean', ascending=False).reset_index(drop=True)
    df.to_csv('outputs/tune_facility_candidates.csv', index=False)

    # Build a small markdown with top-K
    K = min(10, len(df))
    top = df.head(K).copy()
    # Create ready-to-run commands for the best config
    def cmd_from_row(r):
        return ("python -m gqdc.experiments.simulate_facility --config configs/config.yaml --policy all "
                f"--fixed_setpoint {r['fixed_setpoint']} --cool_fraction {r['cool_fraction']} "
                f"--cop_min_at6 {r['cop_min_at6']} --cop_max_at12 {r['cop_max_at12']} "
                f"--ambient_base_c {r['ambient_base_c']} --ambient_swing_c {r['ambient_swing_c']} "
                f"--econ_max_gain {r['econ_max_gain']}")

    top['reduction_mean'] = top['reduction_mean'].map(lambda x: f"{x:.2f}%")
    top['cmd'] = top.apply(cmd_from_row, axis=1)

    with open('outputs/tune_facility_top.md', 'w', encoding='utf-8') as f:
        f.write("# Top parameter sets for facility energy reduction (MPC vs fixed)\n\n")
        f.write(top[['id','reduction_mean','fixed_mean_kwh','mpc_mean_kwh',
                     'fixed_setpoint','cool_fraction','cop_min_at6','cop_max_at12',
                     'ambient_base_c','ambient_swing_c','econ_max_gain','cmd']].to_markdown(index=False))
        f.write("\n")

    print("\nSaved: outputs/tune_facility_candidates.csv")
    print("Saved: outputs/tune_facility_top.md")
    best = df.iloc[0]
    print(f"\nBest≈ params: reduction≈{best['reduction_mean']:.2f}% | fixed≈{best['fixed_mean_kwh']:.2f} kWh → mpc≈{best['mpc_mean_kwh']:.2f} kWh")
    print("Try:\n", cmd_from_row(best))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--candidates', type=int, default=36, help='number of random candidates to evaluate')
    ap.add_argument('--seeds', type=int, default=4, help='number of seeds per candidate')
    ap.add_argument('--seed', type=int, default=0, help='random seed for candidate sampling')
    args = ap.parse_args()
    main(args)
