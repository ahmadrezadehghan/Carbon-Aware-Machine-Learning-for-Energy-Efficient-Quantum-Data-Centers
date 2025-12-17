# Placeholder Bayesian tuner for setpoints
def suggest_next(current_sp, bounds=(6,12), feedback=None):
    step = -0.2 if (feedback is None or feedback.get('energy_trend','down')=='down') else 0.2
    sp = min(max(current_sp + step, bounds[0]), bounds[1])
    return sp
