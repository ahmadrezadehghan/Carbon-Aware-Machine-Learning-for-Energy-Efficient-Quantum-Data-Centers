
# gqdc/experiments/compare_facility.py
import argparse, yaml, numpy as np, pandas as pd
from gqdc.experiments.simulate_facility import make_jobs, simulate_facility_timeline, ambient_signal
from gqdc.simulate.arrivals import generate_arrivals
from gqdc.simulate.carbon import carbon_signal
from gqdc.scheduler.baseline_fifo import run_fifo
from gqdc.quantum.emulator import DEFAULT_WORKLOADS
from gqdc.control.facility_energy import facility_power_kw

def run_once(cfg, seed, fixed_setpoint=6.0, cool_fraction=1.1, cop_min_at6=2.4, cop_max_at12=5.6,
             ambient_base_c=24.0, ambient_swing_c=6.0, econ_max_gain=0.15):
    arr = generate_arrivals(rate_per_min=cfg['signals']['arrivals']['rate_per_min'],
                            diurnal_amp=cfg['signals']['arrivals']['diurnal_amp'],
                            seed=seed)
    carbon = {i: c for i, c in enumerate(
        carbon_signal(base=cfg['signals']['carbon']['base'],
                      swing=cfg['signals']['carbon']['swing'],
                      noise=cfg['signals']['carbon']['noise'],
                      seed=seed))}
    jobs = make_jobs(arr, DEFAULT_WORKLOADS, cfg['experiment']['sla_deadline_s'])
    merged_cfg = cfg | {'carbon_threshold': cfg.get('experiment', {}).get('carbon_threshold', 500.0)}
    sch = run_fifo(jobs, carbon, merged_cfg)
    sch_df = pd.DataFrame(sch)

    fac_params = dict(
        base_it_kw=1.5, heat_per_job_kw=0.4, cryo_kw=3.0,
        cool_fraction=cool_fraction, cop_min_at6=cop_min_at6, cop_max_at12=cop_max_at12,
        ambient_thresh_c=20.0, setpoint_thresh_c=10.0, econ_max_gain=econ_max_gain
    )
    horizon_min = int(np.ceil(sch_df['finish_min'].max())) + 1
    amb = ambient_signal(horizon_min, base_c=ambient_base_c, swing_c=ambient_swing_c)

    # fixed
    ts_rows = []; job_energy = {int(j):0.0 for j in sch_df['job_id'].tolist()}
    for t in range(horizon_min):
        rcount = int((sch_df['start_min'] <= t).sum() - (sch_df['finish_min'] <= t).sum())
        kw = facility_power_kw(rcount, fixed_setpoint, ambient_c=float(amb[t]), **fac_params)
        if rcount>0: job_energy[list(job_energy.keys())[0]] += kw/60.0/rcount  # dummy dist (unused)
        ts_rows.append({'minute': t, 'facility_kw': kw})
    fixed_total = sum(r['facility_kw'] for r in ts_rows)/60.0

    # mpc via simulate_facility_timeline
    ts_m, jd_m, tot_m = simulate_facility_timeline(sch_df, carbon, policy="mpc",
                                                   facility_params=fac_params,
                                                   ambient_base_c=ambient_base_c,
                                                   ambient_swing_c=ambient_swing_c)
    return fixed_total, tot_m['facility_total_kwh']

def main(cfg_path, fixed_setpoint, cool_fraction, cop_min_at6, cop_max_at12, ambient_base_c, ambient_swing_c, econ_max_gain):
    cfg = yaml.safe_load(open(cfg_path))
    seeds = [0,1,2,3,4]
    totals = []
    for s in seeds:
        f, m = run_once(cfg, s, fixed_setpoint, cool_fraction, cop_min_at6, cop_max_at12, ambient_base_c, ambient_swing_c, econ_max_gain)
        totals.append({'seed': s, 'policy':'fixed', 'facility_total_kwh': f})
        totals.append({'seed': s, 'policy':'mpc', 'facility_total_kwh': m})
    df = pd.DataFrame(totals)
    print(df.groupby('policy')['facility_total_kwh'].mean())
    f_mean = df[df['policy']=='fixed']['facility_total_kwh'].mean()
    m_mean = df[df['policy']=='mpc']['facility_total_kwh'].mean()
    red = 100.0*(f_mean - m_mean)/f_mean
    print(f"Average facility energy reduction (mpc vs fixed): {red:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--fixed_setpoint', type=float, default=6.0)
    ap.add_argument('--cool_fraction', type=float, default=1.1)
    ap.add_argument('--cop_min_at6', type=float, default=2.4)
    ap.add_argument('--cop_max_at12', type=float, default=5.6)
    ap.add_argument('--ambient_base_c', type=float, default=24.0)
    ap.add_argument('--ambient_swing_c', type=float, default=6.0)
    ap.add_argument('--econ_max_gain', type=float, default=0.15)
    args = ap.parse_args()
    main(args.config, args.fixed_setpoint, args.cool_fraction, args.cop_min_at6, args.cop_max_at12,
         args.ambient_base_c, args.ambient_swing_c, args.econ_max_gain)
