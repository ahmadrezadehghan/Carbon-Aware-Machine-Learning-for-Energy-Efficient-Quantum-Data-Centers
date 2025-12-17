# RC thermal toy model for cooling dynamics (very simplified)
def step(temp, ambient, power_kw, chilled_sp_c, k=0.05):
    # temp_{t+1} = temp + k*(ambient - temp) + 0.01*power_kw - 0.02*max(0, (12 - chilled_sp_c))
    return temp + k*(ambient - temp) + 0.01*power_kw - 0.02*max(0, (12 - chilled_sp_c))
