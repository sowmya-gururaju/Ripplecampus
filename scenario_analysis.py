"""
scenario_analysis.py - PARALLEL CAMPUS SCENARIOS (A normal / B stress / C stress + interventions),
one-at-a-time intervention impact, and per-factor contribution analysis.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from simulation import SYSTEMS, simulate

DEFAULT_STRESS = dict(attendance_delta=12.0, temp_delta=5.0, rainfall_mm=10.0, event_visitors=1500,
                      event_intensity=2, conc_delta=15.0)
DEFAULT_INTERVENTIONS = dict(stagger_pct=60.0, extra_buses=6, allocation_efficiency_pct=92.0,
                             canteen_capacity_boost_pct=25.0, energy_mgmt_pct=25.0)

FACTOR_GROUPS = {
    "Student population": ["enrolled_students", "intake_growth_pct"],
    "Attendance": ["attendance_rate"],
    "Temperature": ["temperature_c"],
    "Rainfall": ["rainfall_mm"],
    "Timetable": ["simultaneous_classes", "peak_concentration_pct"],
    "Campus event": ["event_day", "event_visitors", "event_intensity"],
}


def intervention_catalog(p):
    """Each intervention applied on its own (kwargs for ScenarioInputs)."""
    return {
        f"Staggered timetable ({p['stagger_pct']:.0f}%)": dict(stagger_pct=p["stagger_pct"]),
        f"More buses (+{int(p['extra_buses'])})": dict(extra_buses=int(p["extra_buses"])),
        f"Better classroom allocation ({p['allocation_efficiency_pct']:.0f}%)":
            dict(allocation_efficiency_pct=p["allocation_efficiency_pct"]),
        f"Canteen capacity +{p['canteen_capacity_boost_pct']:.0f}%":
            dict(canteen_capacity_boost_pct=p["canteen_capacity_boost_pct"]),
        f"Energy management ({p['energy_mgmt_pct']:.0f}%)": dict(energy_mgmt_pct=p["energy_mgmt_pct"]),
    }


def build_scenarios(ref, stress, interventions):
    a = ref
    b = replace(
        a, attendance_rate=min(100.0, a.attendance_rate + stress["attendance_delta"]),
        temperature_c=a.temperature_c + stress["temp_delta"], rainfall_mm=stress["rainfall_mm"],
        peak_concentration_pct=min(95.0, a.peak_concentration_pct + stress["conc_delta"]),
        event_day=True, event_visitors=stress["event_visitors"], event_intensity=int(stress["event_intensity"]))
    c = replace(b, **{k: v for k, v in interventions.items()})
    return {"A - Normal campus": a, "B - Stress day": b, "C - Stress + interventions": c}


def compare(scenarios, A, elec_bundle=None, canteen_bundle=None, fc_elec=None, fc_canteen=None):
    """Metrics x scenarios table. fc_* are forecasting functions (kept out of this module's imports)."""
    results = {n: simulate(i, A) for n, i in scenarios.items()}
    rows = {
        "Electricity utilisation (%)": lambda r, i: r["util"]["Electricity"],
        "Classroom utilisation (%)": lambda r, i: r["util"]["Classrooms"],
        "Canteen demand (meals)": lambda r, i: r["canteen"]["demand"],
        "Canteen utilisation (%)": lambda r, i: r["util"]["Canteen"],
        "Water demand (L)": lambda r, i: r["water"]["demand"],
        "Water utilisation (%)": lambda r, i: r["util"]["Water"],
        "Bus utilisation (%)": lambda r, i: r["util"]["Transport"],
        "Parking utilisation (%)": lambda r, i: r["util"]["Parking"],
        "Congestion index": lambda r, i: r["util"]["Commuting"],
        "Overloaded systems (>100%)": lambda r, i: r["overloaded"],
        "At-risk systems (>=90%)": lambda r, i: r["at_risk"],
    }
    table = pd.DataFrame({n: {k: f(results[n], scenarios[n]) for k, f in rows.items()} for n in scenarios})
    if elec_bundle and fc_elec:
        table.loc["ML forecast: electricity (kWh/day)"] = [fc_elec(elec_bundle, i)["daily_kwh"] for i in scenarios.values()]
    if canteen_bundle and fc_canteen:
        table.loc["ML forecast: meals/day"] = [fc_canteen(canteen_bundle, i) for i in scenarios.values()]
    return table, results


def intervention_impacts(inp_b, A, p):
    """Apply each intervention alone to scenario B; find the bottleneck and the best fix for it."""
    base = simulate(inp_b, A)
    bott = base["bottleneck"]
    rows = []
    for name, kw in intervention_catalog(p).items():
        r = simulate(replace(inp_b, **kw), A)
        row = {"Intervention": name, f"Bottleneck ({bott}) %": round(r["util"][bott], 1),
               "Change (pts)": round(r["util"][bott] - base["util"][bott], 1)}
        for s in SYSTEMS:
            row[s] = round(r["util"][s], 1)
        row["Overloaded"] = r["overloaded"]
        rows.append(row)
    df = pd.DataFrame(rows)
    best = df.loc[df["Change (pts)"].idxmin()]
    return df, bott, best, base


def factor_contributions(inp, ref, A, system):
    """How many utilisation points of `system` each factor adds vs the normal campus."""
    cur = simulate(inp, A)["util"][system]
    out = []
    for name, fields in FACTOR_GROUPS.items():
        reverted = replace(inp, **{f: getattr(ref, f) for f in fields})
        out.append((name, cur - simulate(reverted, A)["util"][system]))
    return sorted(out, key=lambda x: -x[1])
