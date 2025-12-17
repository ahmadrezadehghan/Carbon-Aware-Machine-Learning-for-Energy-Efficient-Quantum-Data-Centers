
# gqdc/experiments/fairness_analysis.py  (WARNING-SAFE: no deprecated groupby.apply on grouping cols)
import argparse, yaml, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gqdc.simulate.arrivals import generate_arrivals
from gqdc.simulate.carbon import carbon_signal
from gqdc.scheduler.baseline_fifo import Job, run_fifo
from gqdc.quantum.emulator import DEFAULT_WORKLOADS

def jain_index(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    s1 = np.sum(x); s2 = np.sum(x**2); n = len(x)
    if n==0 or s2==0: return 1.0
    return float((s1**2)/(n*s2))

def gini_coefficient(x: np.ndarray):
    x = np.asarray(x, dtype=float).flatten()
    if np.amin(x) < 0: x = x - np.amin(x)
    x = np.sort(x); n = len(x)
    if n == 0: return 0.0
    cumx = np.cumsum(x)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0 else 0.0
    return float(gini)

def make_jobs_with_classes(arrivals, workloads, deadline_s_normal, deadline_s_priority, p_priority=0.3):
    rng = np.random.default_rng(0)
    jobs = []; classes = []
    for i, a in enumerate(arrivals):
        w = rng.choice(workloads).name
        is_pri = rng.random() < p_priority
        d_s = deadline_s_priority if is_pri else deadline_s_normal
        jobs.append(Job(i, a, w, d_s))
        classes.append('priority' if is_pri else 'normal')
    return jobs, classes

def apply_deferrals(schedule_df: pd.DataFrame, carbon: dict,
                    deadline_min: pd.Series,
                    forecast_min=60, drop_min=30, deferral_step_min=3, max_deferrals=4,
                    protect_priority=True, classes=None):
    df = schedule_df.copy().reset_index(drop=True)
    df['deadline_min'] = deadline_min.values
    df['deferrals'] = 0; df['wait_min'] = df.get('wait_min', 0.0)
    if classes is not None:
        df['class'] = classes[:len(df)]
    max_t = int(np.ceil(df['finish_min'].max()) + forecast_min + 5)
    c_arr = np.array([carbon.get(t, 300.0) for t in range(max_t)])
    def mean_carbon(s,f):
        s=int(np.floor(s)); f=int(np.ceil(f)); 
        return float(np.mean(c_arr[s:f])) if f>s else c_arr[s]
    for i,row in df.iterrows():
        if protect_priority and classes is not None and df.at[i,'class'] == 'priority':
            continue
        slack = df.at[i,'deadline_min'] - row['finish_min']
        cur_s, cur_f = row['start_min'], row['finish_min']
        d = 0
        while d<max_deferrals and slack>2:
            now = mean_carbon(cur_s, cur_f)
            future = float(np.mean(c_arr[int(cur_f): int(min(cur_f+forecast_min, len(c_arr)-1))]))
            if future <= now - drop_min:
                step = min(deferral_step_min, max(0.0, slack-2))
                if step<=0: break
                cur_s += step; cur_f += step; slack -= step; d+=1
            else:
                break
        if d>0:
            df.at[i,'start_min']=cur_s; df.at[i,'finish_min']=cur_s+(row['finish_min']-row['start_min'])
            df.at[i,'deferrals']=d; df.at[i,'wait_min']=df.at[i,'start_min']-row['arrival_min']
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--p_priority', type=float, default=0.3)
    ap.add_argument('--deadline_normal_s', type=float, default=180.0)
    ap.add_argument('--deadline_priority_s', type=float, default=120.0)
    ap.add_argument('--use_deferrals', action='store_true')
    ap.add_argument('--no_protect_priority', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    arr = generate_arrivals(rate_per_min=cfg.get('signals',{}).get('arrivals',{}).get('rate_per_min',0.9),
                            diurnal_amp=cfg.get('signals',{}).get('arrivals',{}).get('diurnal_amp',0.15))
    c_cfg = cfg.get('signals',{}).get('carbon',{})
    carbon = {i: c for i, c in enumerate(
        carbon_signal(base=c_cfg.get('base',650.0),
                      swing=c_cfg.get('swing',400.0),
                      noise=c_cfg.get('noise',8.0)))}
    jobs, classes = make_jobs_with_classes(arr, DEFAULT_WORKLOADS,
                                           deadline_s_normal=args.deadline_normal_s,
                                           deadline_s_priority=args.deadline_priority_s,
                                           p_priority=args.p_priority)
    sch = run_fifo(jobs, carbon, cfg)
    df = pd.DataFrame(sch)
    df['class'] = classes[:len(df)]
    df['deadline_min'] = df['arrival_min'] + np.where(df['class']=='priority', args.deadline_priority_s/60.0, args.deadline_normal_s/60.0)

    if args.use_deferrals:
        df = apply_deferrals(df, carbon, df['deadline_min'],
                             protect_priority=not args.no_protect_priority,
                             classes=classes)

    # per-class metrics without deprecated behavior
    def metrics_for(sub):
        return pd.Series(dict(
            sla_miss=float((sub['finish_min']>sub['deadline_min']).mean()),
            wait_mean=float(sub['wait_min'].mean()),
            wait_median=float(sub['wait_min'].median())
        ))
    g = df.groupby('class', group_keys=False).apply(metrics_for)

    # fairness (per-job waits)
    waits_all = df['wait_min'].values
    def jain_index_local(x):
        x = np.asarray(x, dtype=float)
        s1 = np.sum(x); s2 = np.sum(x**2); n = len(x)
        if n==0 or s2==0: return 1.0
        return float((s1**2)/(n*s2))
    def gini_coefficient_local(x):
        x = np.asarray(x, dtype=float).flatten()
        if np.amin(x) < 0: x = x - np.amin(x)
        x = np.sort(x); n = len(x)
        if n == 0: return 0.0
        cumx = np.cumsum(x)
        gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0 else 0.0
        return float(gini)
    jain_all = jain_index_local(waits_all)
    gini_all = gini_coefficient_local(waits_all)
    p95_n = df[df['class']=='normal']['wait_min'].quantile(0.95) if (df['class']=='normal').any() else 0.0
    p95_p = df[df['class']=='priority']['wait_min'].quantile(0.95) if (df['class']=='priority').any() else 0.0

    Path('outputs').mkdir(exist_ok=True)
    df.to_csv('outputs/fairness_results.csv', index=False)

    # violin
    plt.figure()
    data = [df[df['class']=='normal']['wait_min'].values, df[df['class']=='priority']['wait_min'].values]
    plt.violinplot(data, showmeans=True)
    plt.xticks([1,2], ['normal','priority'])
    plt.ylabel('Wait (min)'); plt.title(f'Fairness (per-job): Jain={jain_all:.3f}, Gini={gini_all:.3f}')
    plt.tight_layout(); plt.savefig('outputs/fig_fairness_violin.png', dpi=180)

    print("Per-class metrics:\n", g)
    print(f"Per-job fairness — Jain: {jain_all:.3f}, Gini: {gini_all:.3f}, P95(normal): {p95_n:.3f}, P95(priority): {p95_p:.3f}")
    print("Saved: outputs/fairness_results.csv, fig_fairness_violin.png")

if __name__ == '__main__':
    main()
