"""
recommendations.py - rule-based recommendations driven by the actual simulation outputs.

For each system at/above TRIGGER % utilisation the engine:
  1. states what is wrong (numbers from the simulation),
  2. ranks the factors that caused it (revert-one-factor-to-normal analysis),
  3. searches a ladder of interventions for the smallest one that brings it to <= TARGET %,
  4. re-runs the simulation with that intervention to report the estimated improvement
     and any side effects on other systems.
"""
from __future__ import annotations

import math
from dataclasses import replace

from scenario_analysis import factor_contributions
from simulation import SYSTEMS, risk_level, simulate

TRIGGER = 90.0   # recommend at or above this utilisation
TARGET = 85.0    # aim to bring the system to this utilisation or lower


def _ladder(system, inp, A):
    mx = lambda cur, new: max(cur, new)  # noqa: E731  never weaken an intervention already active
    if system == "Electricity":
        return [dict(energy_mgmt_pct=mx(inp.energy_mgmt_pct, e), stagger_pct=mx(inp.stagger_pct, s))
                for e, s in ((10, 0), (15, 30), (15, 60), (25, 60), (30, 80))]
    if system == "Classrooms":
        return [dict(stagger_pct=mx(inp.stagger_pct, s), allocation_efficiency_pct=mx(inp.allocation_efficiency_pct, a))
                for s, a in ((30, 80), (50, 88), (60, 92), (80, 95), (100, 98))]
    if system == "Canteen":
        return [dict(canteen_capacity_boost_pct=mx(inp.canteen_capacity_boost_pct, b)) for b in (10, 20, 30, 40, 60, 80, 100)]
    if system == "Transport":
        return [dict(extra_buses=int(mx(inp.extra_buses, n))) for n in range(1, 41)]
    if system == "Parking":
        return [dict(modal_shift_pct=mx(inp.modal_shift_pct, m)) for m in (10, 20, 30, 40, 50, 60, 80)]
    if system == "Water":
        return [dict(water_saving_pct=mx(inp.water_saving_pct, w)) for w in (5, 10, 15, 20, 30, 40)]
    return []


def _describe(system, kw, inp, A, res):
    if system == "Electricity":
        s = f"Cut cooling load ~{kw['energy_mgmt_pct']:.0f}% (setpoint/cooling schedules, switch off idle rooms)"
        if kw["stagger_pct"] > inp.stagger_pct:
            s += f" and stagger the timetable by ~{kw['stagger_pct']:.0f}% to flatten the peak"
        return s + "."
    if system == "Classrooms":
        return (f"Redistribute classes across periods (stagger ~{kw['stagger_pct']:.0f}% of the peak load) and improve "
                f"room allocation to ~{kw['allocation_efficiency_pct']:.0f}% efficiency (match class size to room size).")
    if system == "Canteen":
        return (f"Raise preparation/serving capacity by ~{kw['canteen_capacity_boost_pct']:.0f}% (extra counters, "
                f"pre-ordering) and stagger serving slots around the peak.")
    if system == "Transport":
        return (f"Add ~{kw['extra_buses'] - inp.extra_buses} bus(es) (total {res['commute']['buses_total'] + kw['extra_buses'] - inp.extra_buses:.0f}) "
                f"or stagger arrival times so fewer riders hit the peak window.")
    if system == "Parking":
        return (f"Encourage public transport/campus buses: shifting ~{kw['modal_shift_pct']:.0f}% of private-vehicle "
                f"commuters to buses, or stagger arrivals, relieves parking.")
    if system == "Water":
        return (f"Reduce consumption ~{kw['water_saving_pct']:.0f}% (leak checks, low-flow fittings, scheduled landscaping) "
                f"or increase supply/storage capacity.")
    return ""


def _what(system, res, u):
    lvl = risk_level(u).lower()
    e, c, k, w, cm = res["elec"], res["classes"], res["canteen"], res["water"], res["commute"]
    if system == "Electricity":
        return (f"Electricity is {lvl} at {u:.0f}%: {e['daily']:,.0f} kWh/day vs {e['capacity_kwh']:,.0f} kWh capacity "
                f"(peak {e['peak_kw']:,.0f} kW vs {e['capacity_kw']:,.0f} kW).")
    if system == "Classrooms":
        return (f"Classroom use is {lvl} at {u:.0f}% in period {c['peak_period']}: ~{c['class_size']:.0f} students per class, "
                f"about {c['overcrowded_rooms']:.0f} rooms overcrowded, {c['unscheduled_sessions']:.0f} sessions without a room.")
    if system == "Canteen":
        return (f"Canteen demand is {lvl} at {u:.0f}%: {k['demand']:,.0f} meals vs {k['capacity']:,.0f} capacity "
                f"({k['excess']:,.0f} meals of excess demand).")
    if system == "Transport":
        return (f"Bus utilisation is {lvl} at {u:.0f}%: {cm['bus_demand_peak']:,.0f} riders in the peak window vs "
                f"{cm['bus_capacity_peak']:,.0f} seats ({cm['buses_total']:.0f} buses).")
    if system == "Parking":
        return (f"Parking is {lvl} at {u:.0f}%: {cm['parking_demand']:,.0f} car-equivalent spaces needed vs "
                f"{cm['parking_capacity']:,.0f} available.")
    return f"Water is {lvl} at {u:.0f}%: {w['demand']:,.0f} L/day vs {w['capacity']:,.0f} L supply."


def generate(inp, res, A, ref):
    recs = []
    for system in sorted(SYSTEMS, key=lambda s: -res["util"][s]):
        u = res["util"][system]
        if u < TRIGGER:
            continue
        drivers = [(n, d) for n, d in factor_contributions(inp, ref, A, system) if d > 0.5][:3]
        chosen, after = None, None
        for kw in _ladder(system, inp, A):
            cand = simulate(replace(inp, **kw), A)
            chosen, after = kw, cand
            if cand["util"][system] <= TARGET:
                break
        rec = dict(system=system, level=risk_level(u), before=u, drivers=drivers,
                   what=_what(system, res, u), action="", after=None, resolved=False, side_effects=[])
        if chosen:
            rec.update(action=_describe(system, chosen, inp, A, res), after=after["util"][system],
                       resolved=after["util"][system] <= TARGET)
            rec["side_effects"] = [f"{s} {res['util'][s]:.0f}% -> {after['util'][s]:.0f}%" for s in SYSTEMS
                                   if s != system and after["util"][s] - res["util"][s] > 3]
            rec["overloaded_after"] = after["overloaded"]
        recs.append(rec)
    return recs
