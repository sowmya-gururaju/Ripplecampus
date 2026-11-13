"""
graph_engine.py - the Butterfly Effect engine.

A NetworkX DiGraph whose edges mirror dependencies that really exist in simulation.py.
* Node state is computed from simulation results (risk colour = utilisation).
* An edge is highlighted when its source AND target both changed versus the "Normal campus".
* verify_edges() proves each factor->system edge by perturbing the factor and re-running
  the simulation, so the graph cannot drift from the maths.
"""
from __future__ import annotations

import math
from dataclasses import replace

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from simulation import SYSTEMS, RISK_COLORS, risk_level, simulate

FACTORS = ["Student population", "Attendance", "Weather", "Timetable", "Campus events"]
POS = {  # fixed layered layout: inputs -> attendance -> commuting -> systems
    "Student population": (0.0, 5.0), "Weather": (0.0, 3.2), "Timetable": (0.0, 1.8), "Campus events": (0.0, 0.2),
    "Attendance": (0.9, 4.4), "Commuting": (1.8, 1.4),
    "Electricity": (2.7, 5.5), "Classrooms": (2.7, 4.4), "Canteen": (2.7, 3.3),
    "Water": (2.7, 2.2), "Parking": (2.7, 1.1), "Transport": (2.7, 0.0),
}
# (source, target, explanation, perturbation key or None for derived/structural)
DEFAULT_EDGES = [
    ("Student population", "Attendance", "Present students = enrolled x attendance rate", "population"),
    ("Student population", "Water", "Hostel residents use water whether or not they attend", "population"),
    ("Attendance", "Classrooms", "More present students -> larger classes and higher seat use", "attendance"),
    ("Attendance", "Electricity", "Occupancy-related load grows with present students", "attendance"),
    ("Attendance", "Canteen", "Meals = present students x participation x meals per student", "attendance"),
    ("Attendance", "Water", "Per-student campus water use", "attendance"),
    ("Attendance", "Commuting", "Non-hostel present students are the commuters", "attendance"),
    ("Weather", "Electricity", "Temperature above threshold drives cooling load", "temperature"),
    ("Weather", "Water", "Heat raises water use", "temperature"),
    ("Weather", "Canteen", "Heat raises beverage/meal demand", "temperature"),
    ("Weather", "Commuting", "Rain shifts two-wheeler/walk users to bus and car", "rainfall"),
    ("Timetable", "Classrooms", "Simultaneous classes and peak concentration set room load", "timetable"),
    ("Timetable", "Electricity", "Sessions add load; concentration raises peak kW", "timetable"),
    ("Timetable", "Commuting", "Concentrated timetables squeeze arrivals into a peak window", "timetable"),
    ("Campus events", "Canteen", "Visitors need meals", "events"),
    ("Campus events", "Electricity", "Event lighting/AV + visitor load", "events"),
    ("Campus events", "Water", "Visitors and catering use water", "events"),
    ("Campus events", "Parking", "Visitor vehicles need parking", "events"),
    ("Campus events", "Transport", "Visitors using bus/shuttle", "events"),
    ("Commuting", "Transport", "Bus share of commuters sets bus demand", None),
    ("Commuting", "Parking", "Car/two-wheeler share of commuters sets parking demand", None),
]


def load_edges(dep_df):
    if dep_df is None or dep_df.empty:
        return DEFAULT_EDGES
    known = {(s, t): (d, p) for s, t, d, p in DEFAULT_EDGES}
    edges = []
    valid = set(FACTORS) | {"Commuting"} | set(SYSTEMS)
    for _, r in dep_df.iterrows():
        s, t = str(r["source"]).strip(), str(r["target"]).strip()
        if s in valid and t in valid and s != t:
            d0, p0 = known.get((s, t), ("", None))
            desc = str(r.get("description", "")) if "description" in dep_df.columns and pd.notna(r.get("description")) else d0
            edges.append((s, t, desc or d0 or "dependency", p0))
    return edges or DEFAULT_EDGES


def build_graph(edges):
    G = nx.DiGraph()
    for n in FACTORS + ["Commuting"] + SYSTEMS:
        G.add_node(n)
    for s, t, d, p in edges:
        G.add_edge(s, t, description=d, perturb=p)
    return G


def _metric(res, node):
    if node == "Attendance":
        return res["pop"]["present"]
    if node in res["util"]:
        return res["util"][node]
    return None


def perturb(inp, key):
    if key == "population":
        return replace(inp, enrolled_students=max(inp.enrolled_students, 100) * 1.2)
    if key == "attendance":
        return replace(inp, attendance_rate=inp.attendance_rate + 10 if inp.attendance_rate <= 90 else inp.attendance_rate - 10)
    if key == "temperature":
        return replace(inp, temperature_c=inp.temperature_c + 8)
    if key == "rainfall":
        return replace(inp, rainfall_mm=inp.rainfall_mm + 20)
    if key == "timetable":
        return replace(inp, simultaneous_classes=math.ceil(max(inp.simultaneous_classes, 1) * 1.25),
                       peak_concentration_pct=min(95, inp.peak_concentration_pct + 15))
    if key == "events":
        return replace(inp, event_day=True, event_visitors=max(inp.event_visitors, 2000))
    return inp


def verify_edges(inp, A, edges):
    """Sensitivity table: perturb each source factor, see whether the target metric moves."""
    base = simulate(inp, A)
    rows = []
    for s, t, d, p in edges:
        if p is None:
            rows.append(dict(Edge=f"{s} -> {t}", Meaning=d, Check="structural (derived value)", Change_pct=None))
            continue
        new = simulate(perturb(inp, p), A)
        b0, b1 = _metric(base, t), _metric(new, t)
        ch = (b1 - b0) / max(abs(b0), 1e-9) * 100
        rows.append(dict(Edge=f"{s} -> {t}", Meaning=d,
                         Check="verified by simulation" if abs(ch) > 1e-6 else "no effect at current settings",
                         Change_pct=round(ch, 2)))
    return pd.DataFrame(rows)


# ------------------------------ node states --------------------------------
def node_states(inp, res, ref_inp, ref_res):
    S = {}
    tot, tot0 = res["pop"]["total"], ref_res["pop"]["total"]
    S["Student population"] = dict(
        kind="factor", text=f"{tot:,.0f} students", delta=f"{(tot - tot0) / max(tot0, 1) * 100:+.1f}% vs normal",
        changed=abs(tot - tot0) > 0.5)
    S["Attendance"] = dict(
        kind="factor", text=f"{inp.attendance_rate:.0f}% ({res['pop']['present']:,.0f} present)",
        delta=f"{inp.attendance_rate - ref_inp.attendance_rate:+.0f} pts vs normal",
        changed=abs(res["pop"]["present"] - ref_res["pop"]["present"]) > 0.5)
    S["Weather"] = dict(
        kind="factor", text=f"{inp.temperature_c:.0f}C, rain {inp.rainfall_mm:.0f} mm",
        delta=f"{inp.temperature_c - ref_inp.temperature_c:+.0f}C, {inp.rainfall_mm - ref_inp.rainfall_mm:+.0f} mm",
        changed=abs(inp.temperature_c - ref_inp.temperature_c) > 0.01 or abs(inp.rainfall_mm - ref_inp.rainfall_mm) > 0.01)
    S["Timetable"] = dict(
        kind="factor", text=f"{inp.simultaneous_classes:.0f} classes/period, {inp.peak_concentration_pct:.0f}% peak",
        delta=f"stagger {inp.stagger_pct:.0f}%",
        changed=abs(res["classes"]["c_eff"] - ref_res["classes"]["c_eff"]) > 1e-6
        or abs(inp.simultaneous_classes - ref_inp.simultaneous_classes) > 0.01)
    S["Campus events"] = dict(
        kind="factor", text=f"Event day, {res['pop']['visitors']:,.0f} visitors" if inp.event_day else "Normal day",
        delta="event active" if inp.event_day else "none", changed=bool(inp.event_day) != bool(ref_inp.event_day)
        or (inp.event_day and abs(inp.event_visitors - ref_inp.event_visitors) > 0.5))
    for n in ["Commuting"] + SYSTEMS:
        u, u0 = res["util"][n], ref_res["util"][n]
        lvl = risk_level(u)
        label = "congestion index" if n == "Commuting" else "utilisation"
        S[n] = dict(kind="system", text=f"{u:.0f}% {label} ({lvl})", delta=f"{u - u0:+.1f} pts vs normal",
                    changed=abs(u - u0) > 0.2, util=u, level=lvl)
    return S


def active_edges(G, S):
    return {(u, v): S[u]["changed"] and S[v]["changed"] for u, v in G.edges}


def ripple_report(G, S):
    """Text bullets explaining each active impact + cascade of every changed factor."""
    act = active_edges(G, S)
    lines = []
    for (u, v), on in act.items():
        if on:
            dv = S[v]["delta"]
            lines.append(f"**{u} -> {v}**: {G.edges[u, v]['description']} ({v}: {dv})")
    cascades = {f: sorted(n for n in nx.descendants(G, f) if S[n]["kind"] == "system" and S[n]["changed"])
                for f in FACTORS if S[f]["changed"]}
    return lines, cascades


def explain_node(G, S, node):
    ins = [f"{u}: {G.edges[u, node]['description']}" for u in G.predecessors(node)]
    outs = [f"{v}: {G.edges[node, v]['description']}" for v in G.successors(node)]
    return ins, outs


# --------------------------------- figure -----------------------------------
def make_figure(G, S):
    act = active_edges(G, S)
    fig = go.Figure()
    for on, color, width in ((False, "rgba(120,135,165,0.35)", 1.2), (True, "#FFB020", 3.2)):
        xs, ys = [], []
        for (u, v), a in act.items():
            if a == on:
                xs += [POS[u][0], POS[v][0], None]
                ys += [POS[u][1], POS[v][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width),
                                 hoverinfo="skip", showlegend=False))
    for (u, v), a in act.items():
        (x0, y0), (x1, y1) = POS[u], POS[v]
        dx, dy = x1 - x0, (y1 - y0)
        L = math.hypot(dx, dy * 0.35) or 1.0
        t = min(0.4, 0.17 / L)
        fig.add_annotation(
            x=x1 - t * dx, y=y1 - t * dy, ax=x0 + t * dx, ay=y0 + t * dy, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=2.6 if a else 1.2,
            arrowcolor="#FFB020" if a else "rgba(120,135,165,0.55)", text="")
    mx = [(POS[u][0] + POS[v][0]) / 2 for u, v in G.edges]
    my = [(POS[u][1] + POS[v][1]) / 2 for u, v in G.edges]
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=14, opacity=0),
                             text=[f"<b>{u} -> {v}</b><br>{G.edges[u, v]['description']}" for u, v in G.edges],
                             hoverinfo="text", showlegend=False))
    names = list(G.nodes)
    colors, lines, sizes, hover = [], [], [], []
    for n in names:
        s = S[n]
        if s["kind"] == "system":
            colors.append(RISK_COLORS[s["level"]])
            lines.append("#FFFFFF" if s["level"] == "Overloaded" else "#0B1220")
        else:
            colors.append("#4DA3FF" if s["changed"] else "#5B6B86")
            lines.append("#0B1220")
        sizes.append(40 if s["kind"] == "system" and s["level"] in ("Critical", "Overloaded") else 32)
        ins, outs = explain_node(G, S, n)
        hover.append(f"<b>{n}</b><br>{s['text']}<br>{s['delta']}<br>"
                     + ("<br>Driven by:<br>" + "<br>".join("- " + i for i in ins) if ins else "")
                     + ("<br>Affects:<br>" + "<br>".join("- " + o for o in outs) if outs else ""))
    fig.add_trace(go.Scatter(
        x=[POS[n][0] for n in names], y=[POS[n][1] for n in names], mode="markers+text", text=names,
        textposition="bottom center", textfont=dict(color="#E6EDF7", size=12),
        marker=dict(size=sizes, color=colors, line=dict(color=lines, width=2)),
        hovertext=hover, hoverinfo="text", showlegend=False))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=620,
        margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False, range=[-0.35, 3.05]),
        yaxis=dict(visible=False, range=[-0.7, 6.1]), hoverlabel=dict(bgcolor="#111A2E"))
    return fig
