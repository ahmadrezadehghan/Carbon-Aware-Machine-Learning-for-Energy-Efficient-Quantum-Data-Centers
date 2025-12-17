
# gqdc/experiments/simulate_facility.py
import argparse, yaml, numpy as np, pandas as pd
from gqdc.simulate.arrivals import generate_arrivals
from gqdc.simulate.carbon import carbon_signal
from gqdc.scheduler.baseline_fifo import Job, run_fifo
from gqdc.quantum.emulator import DEFAULT_WORKLOADS
from gqdc.control.facility_energy import facility_power_kw

def make_jobs(arrivals, workloads, deadline_s):
    rng = np.random.default_rng(0)
    jobs = []
    for i, a in enumerate(arrivals):
        w = rng.choice(workloads).name
        jobs.append(Job(i, a, w, deadline_s))
    return jobs

def build_running_map(schedule_df: pd.DataFrame, horizon_min: int):
    running_map = {t: [] for t in range(horizon_min)}
    for _, row in schedule_df.iterrows():
        s = int(np.floor(row['start_min']))
        f = int(np.ceil(row['finish_min']))
        for t in range(max(0, s), min(horizon_min, f)):
            running_map[t].append(int(row['job_id']))
    return running_map

def ambient_signal(T_minutes: int, base_c: float = 24.0, swing_c: float = 6.0, phase: float = 0.0):
    # Simple diurnal ambient temperature (°C): base ± swing
    t = np.arange(T_minutes)
    return base_c + swing_c * np.sin(2*np.pi*(t/1440.0 + phase))

def choose_setpoint_mpc(t: int, prev_sp: float, running_map: dict, carbon: dict,
                        ambient: np.ndarray,
                        horizon: int = 30, ramp_limit_c: float = 2.0,
                        min_c: float = 6.0, max_c: float = 12.0,
                        high_load_thresh: int = 4,
                        facility_params: dict = None):
    fac = facility_params or {}
    candidates = [min(max(prev_sp + d, min_c), max_c) for d in np.linspace(-ramp_limit_c, ramp_limit_c, 9)]
    if prev_sp not in candidates: candidates.append(prev_sp)
    best_sp, best_cost = prev_sp, float('inf')
    for sp in candidates:
        cost = 0.0
        for k in range(horizon):
            tt = t + k
            if tt >= len(ambient):
                break
            rcount = len(running_map.get(tt, []))
            amb = float(ambient[tt])
            kw = facility_power_kw(rcount, sp, ambient_c=amb, **fac)
            if rcount >= high_load_thresh and sp > 8.0:
                kw *= (1.0 + 0.1*(sp - 8.0))  # stronger penalty under heavy load
            cost += kw
        if cost < best_cost:
            best_cost, best_sp = cost, sp
    return float(best_sp)

def simulate_facility_timeline(schedule_df: pd.DataFrame,
                               carbon: dict,
                               policy: str = "fixed",
                               fixed_setpoint_c: float = 6.5,
                               facility_params: dict = None,
                               ambient_base_c: float = 24.0,
                               ambient_swing_c: float = 6.0):
    facility_params = facility_params or {}
    horizon_min = int(np.ceil(schedule_df['finish_min'].max())) + 1
    running_map = build_running_map(schedule_df, horizon_min)
    amb = ambient_signal(horizon_min, base_c=ambient_base_c, swing_c=ambient_swing_c)

    # simple rule-based (legacy)
    def smart_setpoint_controller(prev_setpoint: float, running_jobs: int, carbon_now: float):
        sp = prev_setpoint
        if running_jobs <= 2:
            sp = min(12.0, sp + 1.0)
        elif running_jobs >= 4:
            sp = max(6.0, sp - 1.0)
        else:
            target = 9.5
            if sp < target: sp = min(12.0, sp + 0.5)
            elif sp > target: sp = max(6.0, sp - 0.5)
        return float(sp)

    setpoint = fixed_setpoint_c if policy == "fixed" else 9.0
    ts_rows = []
    job_energy = {int(j): 0.0 for j in schedule_df['job_id'].tolist()}
    for t in range(horizon_min):
        rjobs = running_map[t]
        rcount = len(rjobs)
        if policy == "fixed":
            sp = setpoint
        elif policy == "smart":
            sp = smart_setpoint_controller(setpoint, rcount, 0.0)
        else:  # 'mpc'
            sp = choose_setpoint_mpc(t, setpoint, running_map, {}, amb, facility_params=facility_params)
        kw = facility_power_kw(rcount, sp, ambient_c=float(amb[t]), **facility_params)
        kwh = kw / 60.0
        if rcount > 0:
            share = kwh / rcount
            for jid in rjobs:
                job_energy[jid] += share
        ts_rows.append({'minute': t, 'running_jobs': rcount, 'setpoint_c': sp, 'facility_kw': kw, 'ambient_c': float(amb[t])})
        setpoint = sp

    ts_df = pd.DataFrame(ts_rows)
    job_rows = []
    for _, row in schedule_df.iterrows():
        jid = int(row['job_id'])
        job_rows.append({
            'job_id': jid,
            'energy_fac_kwh': job_energy.get(jid, 0.0),
            'start_min': row['start_min'],
            'finish_min': row['finish_min'],
        })
    job_df = pd.DataFrame(job_rows)
    totals = {
        'facility_total_kwh': ts_df['facility_kw'].sum()/60.0,
        'facility_per_job_kwh_mean': job_df['energy_fac_kwh'].mean()
    }
    return ts_df, job_df, totals

def main(cfg_path, policy, fixed_setpoint, cool_fraction, cop_min_at6, cop_max_at12,
         ambient_base_c, ambient_swing_c, econ_max_gain):
    cfg = yaml.safe_load(open(cfg_path))
    arr = generate_arrivals(rate_per_min=cfg['signals']['arrivals']['rate_per_min'],
                            diurnal_amp=cfg['signals']['arrivals']['diurnal_amp'])
    carbon = {i: c for i, c in enumerate(
        carbon_signal(base=cfg['signals']['carbon']['base'],
                      swing=cfg['signals']['carbon']['swing'],
                      noise=cfg['signals']['carbon']['noise']))}
    jobs = make_jobs(arr, DEFAULT_WORKLOADS, cfg['experiment']['sla_deadline_s'])
    merged_cfg = cfg | {'carbon_threshold': cfg.get('experiment', {}).get('carbon_threshold', 1e9)}
    schedule = run_fifo(jobs, carbon, merged_cfg)
    sch_df = pd.DataFrame(schedule)

    fac_params = dict(
        base_it_kw=1.5,
        heat_per_job_kw=0.4,
        cryo_kw=3.0,
        cool_fraction=cool_fraction,
        cop_min_at6=cop_min_at6,
        cop_max_at12=cop_max_at12,
        ambient_thresh_c=20.0,
        setpoint_thresh_c=10.0,
        econ_max_gain=econ_max_gain
    )

    results = []
    if policy in ("fixed","both","all"):
        ts, jd, totals = simulate_facility_timeline(sch_df, carbon, policy="fixed",
                                                    fixed_setpoint_c=fixed_setpoint,
                                                    facility_params=fac_params,
                                                    ambient_base_c=ambient_base_c,
                                                    ambient_swing_c=ambient_swing_c)
        ts.to_csv("outputs/facility_ts_fixed.csv", index=False)
        jd.to_csv("outputs/facility_jobs_fixed.csv", index=False)
        results.append(('fixed', totals))
    if policy in ("smart","both","all"):
        ts, jd, totals = simulate_facility_timeline(sch_df, carbon, policy="smart",
                                                    facility_params=fac_params,
                                                    ambient_base_c=ambient_base_c,
                                                    ambient_swing_c=ambient_swing_c)
        ts.to_csv("outputs/facility_ts_smart.csv", index=False)
        jd.to_csv("outputs/facility_jobs_smart.csv", index=False)
        results.append(('smart', totals))
    if policy in ("mpc","both","all"):
        ts, jd, totals = simulate_facility_timeline(sch_df, carbon, policy="mpc",
                                                    facility_params=fac_params,
                                                    ambient_base_c=ambient_base_c,
                                                    ambient_swing_c=ambient_swing_c)
        ts.to_csv("outputs/facility_ts_mpc.csv", index=False)
        jd.to_csv("outputs/facility_jobs_mpc.csv", index=False)
        results.append(('mpc', totals))

    for name, tot in results:
        print(f"[{name}] facility_total_kwh={tot['facility_total_kwh']:.3f} | per_job_mean_kwh={tot['facility_per_job_kwh_mean']:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--policy', choices=['fixed','smart','mpc','both','all'], default='all')
    ap.add_argument('--fixed_setpoint', type=float, default=6.0)
    ap.add_argument('--cool_fraction', type=float, default=1.1)
    ap.add_argument('--cop_min_at6', type=float, default=2.4)
    ap.add_argument('--cop_max_at12', type=float, default=5.6)
    ap.add_argument('--ambient_base_c', type=float, default=24.0)
    ap.add_argument('--ambient_swing_c', type=float, default=6.0)
    ap.add_argument('--econ_max_gain', type=float, default=0.15)
    args = ap.parse_args()
    main(args.config, args.policy, args.fixed_setpoint, args.cool_fraction, args.cop_min_at6, args.cop_max_at12,
         args.ambient_base_c, args.ambient_swing_c, args.econ_max_gain)
