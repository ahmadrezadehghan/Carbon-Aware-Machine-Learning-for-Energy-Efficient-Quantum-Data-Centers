
# gqdc/control/facility_energy.py
# Facility energy with ambient-aware cooling and optional economizer.
# Goal: allow larger savings when setpoint is high and ambient is cool.

def cop_from_setpoint(setpoint_c: float, min_at6: float = 2.4, max_at12: float = 5.6) -> float:
    """
    Linear COP(setpoint) between 6°C and 12°C.
    Default max_at12 slightly higher than Patch 5 for stronger sensitivity.
    """
    if setpoint_c <= 6.0:
        return min_at6
    if setpoint_c >= 12.0:
        return max_at12
    slope = (max_at12 - min_at6) / 6.0
    return min_at6 + slope * (setpoint_c - 6.0)

def economizer_gain(ambient_c: float, setpoint_c: float,
                    ambient_thresh_c: float = 20.0,
                    setpoint_thresh_c: float = 10.0,
                    econ_max_gain: float = 0.15) -> float:
    """
    When ambient is cool (below threshold) and setpoint is high (>= threshold),
    an air-side/water-side economizer can offset part of cooling power.
    Return fractional reduction [0..econ_max_gain].
    """
    if ambient_c >= ambient_thresh_c or setpoint_c < setpoint_thresh_c:
        return 0.0
    # Cooler ambient => larger gain up to econ_max_gain
    # Example: ambient 15C -> ~ (20-15)/10 = 0.5 of max; ambient 10C -> full max.
    frac = min(1.0, (ambient_thresh_c - ambient_c) / 10.0)
    return econ_max_gain * frac

def facility_power_kw(running_jobs: int,
                      setpoint_c: float,
                      ambient_c: float = 24.0,
                      base_it_kw: float = 1.5,
                      heat_per_job_kw: float = 0.4,
                      cryo_kw: float = 3.0,
                      cool_fraction: float = 1.1,
                      cop_min_at6: float = 2.4,
                      cop_max_at12: float = 5.6,
                      ambient_thresh_c: float = 20.0,
                      setpoint_thresh_c: float = 10.0,
                      econ_max_gain: float = 0.15) -> float:
    """
    P_total = IT + Cryo + Cooling
    Cooling = (cool_fraction * IT) / COP(setpoint) * (1 - economizer_gain)
    """
    it_kw = base_it_kw + heat_per_job_kw * max(0, running_jobs)
    cop = cop_from_setpoint(setpoint_c, min_at6=cop_min_at6, max_at12=cop_max_at12)
    econ = economizer_gain(ambient_c, setpoint_c,
                           ambient_thresh_c=ambient_thresh_c,
                           setpoint_thresh_c=setpoint_thresh_c,
                           econ_max_gain=econ_max_gain)
    cooling_kw = (cool_fraction * it_kw) / cop
    cooling_kw *= (1.0 - econ)
    total_kw = it_kw + cryo_kw + cooling_kw
    return float(total_kw)
