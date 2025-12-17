
# gqdc/control/facility_energy_nl.py
"""
Nonlinear COP model with partial-load effect for cooling.
- COP increases with setpoint and with part-load (diminishing returns).
- Use as a drop-in alternative to facility_energy.facility_power_kw.
"""
import math

def cop_nonlinear(setpoint_c: float, load_frac: float,
                  base_cop_at6: float = 2.3, base_cop_at12: float = 5.6,
                  alpha_load: float = 0.6):
    """
    Piecewise-smooth COP that depends on setpoint and partial load (0..1).
    - Linear in setpoint baseline between 6 and 12 C.
    - Multiplied by (1 + alpha_load*(1 - load_frac)) to represent better efficiency at lower loads.
    """
    sp = max(6.0, min(12.0, setpoint_c))
    slope = (base_cop_at12 - base_cop_at6) / 6.0
    cop_sp = base_cop_at6 + slope * (sp - 6.0)
    lf = max(0.0, min(1.0, load_frac))
    return float(cop_sp * (1.0 + alpha_load*(1.0 - lf)))

def facility_power_kw_nl(running_jobs: int,
                         setpoint_c: float,
                         ambient_c: float = 24.0,
                         base_it_kw: float = 1.5,
                         heat_per_job_kw: float = 0.4,
                         cryo_kw: float = 3.0,
                         cool_fraction: float = 1.1,
                         econ_max_gain: float = 0.15,
                         ambient_thresh_c: float = 20.0,
                         setpoint_thresh_c: float = 10.0,
                         base_cop_at6: float = 2.3,
                         base_cop_at12: float = 5.6,
                         alpha_load: float = 0.6,
                         it_kw_nominal: float = 3.0):
    """
    Total power = IT + Cryo + Cooling_nl
    where Cooling_nl = (cool_fraction * IT) / COP_nonlinear(setpoint, load_frac) * (1 - economizer_gain)
    load_frac = IT / (IT_nominal + eps)
    """
    it_kw = base_it_kw + heat_per_job_kw * max(0, running_jobs)
    # simple economizer gain
    econ = 0.0
    if ambient_c < ambient_thresh_c and setpoint_c >= setpoint_thresh_c:
        econ = min(econ_max_gain, (ambient_thresh_c - ambient_c)/10.0 * econ_max_gain)
    load_frac = min(1.0, max(0.05, it_kw / max(it_kw_nominal, 0.1)))
    cop = cop_nonlinear(setpoint_c, load_frac, base_cop_at6, base_cop_at12, alpha_load)
    cooling_kw = (cool_fraction * it_kw) / max(cop, 0.1)
    cooling_kw *= (1.0 - econ)
    total_kw = it_kw + cryo_kw + cooling_kw
    return float(total_kw)
