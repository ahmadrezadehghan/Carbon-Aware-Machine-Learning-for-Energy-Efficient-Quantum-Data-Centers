
# gqdc/experiments/compare_emissions_aggressive.py
"""
Aggressive carbon-aware deferral demo (v2)
- Post-process FIFO schedule within SLA slack (or overridden deadline).
- Flexible rule: either strict threshold OR "future carbon lower by drop_min" (no threshold).
- Optional --deadline_s to increase slack only for this experiment (no need to edit config).
- Prints emissions reduction and deferrals/job.

Usage example (Windows CMD):
  python -m gqdc.experiments.compare_emissions_aggressive --config configs/config.yaml ^
    --carbon_base 650 --carbon_swing 400 --threshold 480 --strict_threshold ^
    --forecast 60 --drop_min 40 --deferral_step_min 4 --max_deferrals 8 --deadline_s 5400 --seeds 5
"""
import argparse, yaml, numpy as np, pandas as pd
from pathlib import Path
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

def mean_carbon_over_interval(carbon: dict, start: float, finish: float) -> float:
    s = int(np.floor(start)); f = int(np.ceil(finish))
    if f <= s:
        return float(carbon.get(s, 300.0))
    vals = [carbon.get(t, 300.0) for t in range(s, f)]
    return float(np.mean(vals)) if len(vals) else 300.0

def apply_deferrals(schedule_df: pd.DataFrame, carbon: dict,
                    deadline_s: float,
                    threshold: float = 450.0,
                    strict_threshold: bool = False,
                    forecast_min: int = 30,
                    drop_min: float = 40.0,
                    guard_min: float = 2.0,
                    deferral_step_min: int = 2,
                    max_deferrals: int = 5):
    df = schedule_df.copy().sort_values('arrival_min').reset_index(drop=True)
    if 'deadline_min' not in df.columns:
        df['deadline_min'] = df['arrival_min'] + (deadline_s/60.0)
    df['deferrals'] = 0
    df['wait_min'] = df.get('wait_min', 0.0)

    max_t = int(np.ceil(df['finish_min'].max()) + forecast_min + 5)
    c_arr = np.array([carbon.get(t, 300.0) for t in range(max_t)])

    for i,row in df.iterrows():
        start = row['start_min']; finish = row['finish_min']
        run_min = finish - start
        slack = df.at[i, 'deadline_min'] - finish
        d = 0
        cur_start = start; cur_finish = finish
        while d < max_deferrals:
            if slack <= guard_min:
                break
            now_c = mean_carbon_over_interval(carbon, cur_start, cur_finish)
            # forecast over next window
            f_lo = int(cur_finish)
            f_hi = int(min(cur_finish + forecast_min, len(c_arr)-1))
            if f_hi <= f_lo:
                break
            future_c = float(np.mean(c_arr[f_lo:f_hi]))
            # decision
            ok = False
            if strict_threshold:
                ok = (now_c >= threshold) and (future_c < threshold) and ((now_c - future_c) >= drop_min)
            else:
                ok = (future_c <= now_c - drop_min)  # pure "lower-by" rule
            if ok:
                step = min(deferral_step_min, max(0.0, slack - guard_min))
                if step <= 0: break
                cur_start += step; cur_finish += step
                slack -= step; d += 1
            else:
                break

        if d>0:
            df.at[i,'start_min'] = cur_start
            df.at[i,'finish_min'] = cur_finish
            df.at[i,'deferrals'] = d
            df.at[i,'wait_min'] = df.at[i,'start_min'] - df.at[i,'arrival_min']
            if 'energy_proxy' in df.columns:
                mcarb = mean_carbon_over_interval(carbon, cur_start, cur_finish)
                df.at[i,'emissions_index'] = float(df.at[i,'energy_proxy']) * mcarb

    return df

def run_seed(cfg, seed, args):
    arr = generate_arrivals(rate_per_min=cfg['signals']['arrivals']['rate_per_min'],
                            diurnal_amp=cfg['signals']['arrivals']['diurnal_amp'],
                            seed=seed)
    carbon = {i: c for i, c in enumerate(
        carbon_signal(base=args.carbon_base, swing=args.carbon_swing,
                      noise=cfg['signals']['carbon']['noise'], seed=seed))}
    # deadline override
    d_s = args.deadline_s if args.deadline_s is not None else cfg['experiment']['sla_deadline_s']
    jobs = make_jobs(arr, DEFAULT_WORKLOADS, d_s)
    merged_cfg = cfg | {'carbon_threshold': args.threshold}
    sch = run_fifo(jobs, carbon, merged_cfg)
    dfA = pd.DataFrame(sch); dfA['label'] = 'A_blind'; dfA['seed'] = seed
    dfB = apply_deferrals(dfA, carbon,
                          deadline_s=d_s,
                          threshold=args.threshold,
                          strict_threshold=args.strict_threshold,
                          forecast_min=args.forecast,
                          drop_min=args.drop_min,
                          guard_min=args.guard_min,
                          deferral_step_min=args.deferral_step_min,
                          max_deferrals=args.max_deferrals).copy()
    dfB['label'] = 'B_carbon_smart'; dfB['seed'] = seed
    return dfA, dfB

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--carbon_base', type=float, default=650.0)
    ap.add_argument('--carbon_swing', type=float, default=400.0)
    ap.add_argument('--threshold', type=float, default=480.0)
    ap.add_argument('--strict_threshold', action='store_true', help='if set, require future < threshold and drop>=drop_min')
    ap.add_argument('--forecast', type=int, default=60)
    ap.add_argument('--drop_min', type=float, default=40.0)
    ap.add_argument('--guard_min', type=float, default=2.0)
    ap.add_argument('--deferral_step_min', type=int, default=4)
    ap.add_argument('--max_deferrals', type=int, default=8)
    ap.add_argument('--deadline_s', type=float, default=None, help='override SLA deadline (seconds) for this experiment')
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))

    frames = []
    for s in range(args.seeds):
        A, B = run_seed(cfg, s, args)
        frames.append(A); frames.append(B)

    out = pd.concat(frames, ignore_index=True)
    Path('outputs').mkdir(exist_ok=True)
    out.to_csv('outputs/emissions_aggressive.csv', index=False)

    g = out.groupby('label')['emissions_index'].mean()
    d = out.groupby('label')['deferrals'].mean()
    e_red = 100.0*(g['A_blind'] - g['B_carbon_smart'])/g['A_blind']
    print("Emissions mean (A blind):", g['A_blind'])
    print("Emissions mean (B smart):", g['B_carbon_smart'])
    print(f"Emissions reduction (smart vs blind): {e_red:.2f}%")
    print("Avg deferrals per job:", d.to_dict())

if __name__ == '__main__':
    main()
