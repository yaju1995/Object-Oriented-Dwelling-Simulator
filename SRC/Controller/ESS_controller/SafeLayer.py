import numpy as np


def soc_safety_layer(u, soc, soc_min=0.05, soc_max=1, dt_hours=1, capacity_Wh=10, eps=1e-12):
    """
    Safety layer for battery SoC using Dalal et al. linear projection.

    u : float
        Raw action (positive = charge, negative = discharge)
    soc : float
        Current state of charge [0,1]
    soc_min, soc_max : float
        Allowed SoC bounds
    dt_hours : float
        Timestep in hours
    capacity_Wh : float
        Battery capacity in Wh
    """
    k = dt_hours / capacity_Wh  # SOC change per unit action

    # Two constraints: upper and lower
    constraints = []

    # Upper constraint: SOC_t + k*u <= SOC_max
    h_upper = soc - soc_max
    g_upper = k
    constraints.append((h_upper, g_upper))

    # Lower constraint: SOC_t + k*u >= SOC_min  ->  -k*u + (soc_min - soc) <= 0
    h_lower = soc_min - soc
    g_lower = -k
    constraints.append((h_lower, g_lower))

    # Evaluate violations
    worst_violation = 0
    worst = None

    for h_s, g_s in constraints:
        violation = g_s * u + h_s
        if violation > worst_violation:
            worst_violation = violation
            worst = (h_s, g_s)

    # No violation → safe
    if worst is None:
        return u

    h_s, g_s = worst
    g_norm_sq = g_s * g_s

    if g_norm_sq < eps:
        return u  # cannot correct

    lam = worst_violation / g_norm_sq
    a_safe = u - lam * g_s
    return a_safe
