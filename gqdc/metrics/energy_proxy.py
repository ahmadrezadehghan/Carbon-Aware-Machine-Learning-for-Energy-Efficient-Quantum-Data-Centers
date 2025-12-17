def energy_proxy(g1q:int, g2q:int, meas:int, runtime_s:float, alpha_1q=1.0, beta_2q=5.0, gamma_meas=1.0, overhead_per_min=3.0):
    return alpha_1q*g1q + beta_2q*g2q + gamma_meas*meas + overhead_per_min*(runtime_s/60.0)
