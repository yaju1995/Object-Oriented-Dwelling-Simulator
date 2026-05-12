import pandas as pd
import numpy as np


def extract_controller_metrics(controller, ev_capacity_Wh):
    ev = controller.ev_controller
    db = controller.hems_database.df

    metrics = {
        "unsatisfied_soc": ev.unsatified_energy,
        "satisfied_soc": ev.satisfied_energy,
        "unsatisfied_soc_Wh": ev.unsatified_energy * ev_capacity_Wh,
        "satisfied_soc_Wh": ev.satisfied_energy * ev_capacity_Wh,
        "not_full_count": ev.not_full_count,
        "ev_charging_cost": ev.total_ev_charging_cost,
        "ev_charging_energy": ev.total_ev_charging_energy,
        "ev_cost_per_kWh": (
            ev.total_ev_charging_cost / ev.total_ev_charging_energy
            if ev.total_ev_charging_energy > 0 else 0
        ),
        "final_house_cost": db["Instant Cost"].sum()
    }

    return metrics


def extract_ev_connection_metrics(df_1):
    df = df_1.copy()

    # Detect transitions
    df["parked_shift"] = df["EV Parked"].shift(1)

    df["connect_event"] = (df["EV Parked"] == True) & (df["parked_shift"] == False)
    df["disconnect_event_raw"] = (df["EV Parked"] == False) & (df["parked_shift"] == True)

    # SOC at connection
    connections = df.loc[df["connect_event"], "EV SOC (-)"]

    # SOC at disconnection (use shift instead of subtracting time)
    disconnections = df["EV SOC (-)"].shift(1).loc[df["disconnect_event_raw"]]

    return {
        "connect_soc_list": list(connections.values),
        "disconnect_soc_list": list(disconnections.values),
        "avg_connect_soc": connections.mean() if len(connections) else None,
        "median_disconnect_soc": disconnections.median() if len(disconnections) else None,
        "num_connections": len(connections),
        "num_disconnections": len(disconnections)
    }
