"""MIRROR - Smart Campus Ripple Analytics (Streamlit UI). Run: streamlit run app.py"""
from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_loader as dl
import forecasting as fc
import graph_engine as ge
import recommendations as rec
import scenario_analysis as sa
from simulation import DAYS, DEFAULT_ASSUMPTIONS, RISK_COLORS, SYSTEMS, risk_color, risk_level, simulate

st.set_page_config(page_title="MIRROR - Smart Campus Ripple Analytics", page_icon="🪞", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
.hero h1 {margin: 0; font-size: 2.1rem; letter-spacing: .5px;}
.hero p {margin: 0; color: #8FA3C4;}
.card {background:#111A2E; border:1px solid #1E2A44; border-radius:12px; padding:14px 16px; margin-bottom:10px;}
.card .t {color:#8FA3C4; font-size:.80rem; text-transform:uppercase; letter-spacing:.8px;}
.card .v {font-size:1.8rem; font-weight:700; line-height:1.25;}
.card .s {color:#8FA3C4; font-size:.80rem;}
.pill {display:inline-block; padding:1px 10px; border-radius:999px; font-size:.75rem; font-weight:600; color:#0B1220;}
.note {color:#8FA3C4; font-size:.85rem;}
</style>""", unsafe_allow_html=True)


# ------------------------------ UI helpers --------------------------------
def _stretch(fn):
    """Version-proof 'fill the container' kwarg (width='stretch' on new Streamlit, else use_container_width)."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


def show(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, **_stretch(st.plotly_chart))


def table(df, **kw):
    st.dataframe(df, hide_index=kw.get("hide_index", True), **_stretch(st.dataframe))


def card(title, value, sub="", util=None):
    color = risk_color(util) if util is not None else "#4DA3FF"
    pill = f'<span class="pill" style="background:{color}">{risk_level(util)}</span>' if util is not None else ""
    st.markdown(f'<div class="card" style="border-left:5px solid {color}"><div class="t">{title}</div>'
                f'<div class="v">{value}</div><div class="s">{sub} {pill}</div></div>', unsafe_allow_html=True)


def util_bar(labels, values, title="", height=320):
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=[risk_color(v) for v in values],
                           text=[f"{v:.0f}%" for v in values], textposition="outside"))
    for y, c in ((70, RISK_COLORS["Warning"]), (90, RISK_COLORS["Critical"]), (100, RISK_COLORS["Overloaded"])):
        fig.add_hline(y=y, line_dash="dot", line_color=c, opacity=.6)
    fig.update_layout(title=title, height=height, yaxis_title="Utilisation %",
                      yaxis_range=[0, max(120, max(values) * 1.15)])
    show(fig)


# ------------------------------- data layer --------------------------------
@st.cache_data(show_spinner="Loading datasets...")
def cached_load(upload_items):
    return dl.load_all(upload_items)


@st.cache_resource(show_spinner="Training forecasting models...")
def train_models(elec_df, canteen_df):
    out = {"elec": None, "canteen": None, "errors": []}
    for key, df, fn in (("elec", elec_df, fc.train_electricity), ("canteen", canteen_df, fc.train_canteen)):
        if df is not None:
            try:
                out[key] = fn(df)
            except Exception as exc:
                out["errors"].append(f"{key}: {exc}")
    return out


st.markdown('<div class="hero"><h1>🪞 MIRROR</h1><p><b>Smart Campus Ripple Analytics</b> - '
            'One campus. Many decisions. Endless ripple effects.</p></div>', unsafe_allow_html=True)

# ---- sidebar: data
sb = st.sidebar
sb.title("Scenario controls")
uploads = []
with sb.expander("📂 Data & CSV upload"):
    st.caption("Uploads override files in data/. Use the standard dataset names.")
    for name in dl.DATASETS:
        f = st.file_uploader(name + ".csv", type="csv", key=f"up_{name}")
        if f is not None:
            uploads.append((name, f.getvalue()))
    synthetic = st.checkbox("My datasets are synthetic / demo data", value=True,
                            help="Labels forecasts and evaluation as synthetic-data demonstrations.")
data, report = cached_load(tuple(uploads))
A0, notes = dl.build_assumptions(data)
ref0 = dl.reference_inputs(A0)
version = hash(tuple((n, len(b)) for n, b in uploads))

# ---- main: configurable assumptions
with st.expander("⚙️ Configurable assumptions (every formula uses these) & data status"):
    st.markdown("**Dataset status**")
    table(pd.DataFrame(report))
    if notes:
        st.caption("Calibration: " + " | ".join(notes[:12]) + (" ..." if len(notes) > 12 else ""))
    adf = pd.DataFrame({"key": list(A0), "value": [float(v) for v in A0.values()],
                        "description": [DEFAULT_ASSUMPTIONS[k][1] for k in A0]})
    edited = st.data_editor(adf, hide_index=True, disabled=["key", "description"], key=f"assump_{version}",
                            height=380)
    A_user = {r.key: (float(r.value) if pd.notna(r.value) else A0[r.key]) for r in edited.itertuples()}
A = dl.finalize_assumptions(A_user)
ref = dl.reference_inputs(A)

# ---- sidebar: inputs
def _reset():
    for k in [k for k in st.session_state if k.startswith("in_")]:
        del st.session_state[k]


sb.button("↺ Reset to normal campus", on_click=_reset)
sb.subheader("🎓 Student population")
enrolled = sb.number_input("Enrolled students", 100, 200000, int(ref.enrolled_students), 100, key="in_enrolled")
growth = sb.slider("Intake growth (%)", -20.0, 50.0, 0.0, 0.5, key="in_growth")
sb.subheader("📅 Attendance")
attendance = sb.slider("Attendance rate (%)", 20.0, 100.0, float(np.clip(ref.attendance_rate, 20, 100)), 1.0, key="in_att")
sb.subheader("🌦️ Weather")
temperature = sb.slider("Temperature (C)", 5.0, 48.0, float(np.clip(ref.temperature_c, 5, 48)), 0.5, key="in_temp")
rainfall = sb.slider("Rainfall (mm)", 0.0, 80.0, 0.0, 1.0, key="in_rain")
sb.subheader("🕘 Timetable")
max_sim = int(max(10, A["total_classrooms"] * 1.5))
simul = sb.slider("Classes per period", 1, max_sim, int(np.clip(ref.simultaneous_classes, 1, max_sim)), key="in_sim")
conc = sb.slider("Classes in peak periods (%)", 10.0, 95.0, float(np.clip(ref.peak_concentration_pct, 10, 95)), 1.0, key="in_conc")
sb.subheader("🎉 Campus events")
event_day = sb.checkbox("Event day", False, key="in_event")
visitors = sb.number_input("Event visitors", 0, 100000, 1500, 100, key="in_vis", disabled=not event_day)
intensity = sb.select_slider("Event intensity", [1, 2, 3], 2, format_func=lambda x: {1: "Low", 2: "Medium", 3: "High"}[x],
                             key="in_int", disabled=not event_day)
dow = sb.selectbox("Day of week (used by ML forecasts)", range(7), 2, format_func=lambda i: DAYS[i], key="in_dow")
with sb.expander("🛠️ Interventions (optional)"):
    st.caption("Commuting is an output - these levers change its drivers/capacity.")
    stagger = st.slider("Staggered timetable (%)", 0, 100, 0, key="in_stagger")
    alloc = st.slider("Classroom allocation efficiency (%)", 50, 100, int(A["allocation_efficiency_pct"]), key="in_alloc")
    extra_buses = st.slider("Extra buses", 0, 40, 0, key="in_buses")
    canteen_boost = st.slider("Canteen capacity boost (%)", 0, 100, 0, key="in_canteen")
    energy_mgmt = st.slider("Electricity demand management (%)", 0, 50, 0, key="in_energy")
    modal_shift = st.slider("Shift private vehicles to bus (%)", 0, 80, 0, key="in_modal")
    water_saving = st.slider("Water saving (%)", 0, 40, 0, key="in_water")

inp = replace(ref, enrolled_students=enrolled, intake_growth_pct=growth, attendance_rate=attendance,
                 temperature_c=temperature, rainfall_mm=rainfall, simultaneous_classes=simul,
                 peak_concentration_pct=conc, event_day=event_day, event_visitors=visitors, event_intensity=intensity,
                 day_of_week=dow, stagger_pct=stagger, allocation_efficiency_pct=alloc, extra_buses=extra_buses,
                 canteen_capacity_boost_pct=canteen_boost, energy_mgmt_pct=energy_mgmt, modal_shift_pct=modal_shift,
                 water_saving_pct=water_saving)

res = simulate(inp, A)
res_ref = simulate(ref, A)
models = train_models(data.get("electricity_data"), data.get("canteen_data"))
em, cm_ = models["elec"], models["canteen"]
if models["errors"]:
    st.warning("Forecast model issue: " + "; ".join(models["errors"]))
tag = " (synthetic-data demonstration)" if synthetic else ""

tabs = st.tabs(["Overview", "⚡ Electricity", "🏫 Classrooms", "🍽️ Canteen", "🚌 Commuting & Parking", "💧 Water",
                "🦋 Butterfly Graph", "📈 Forecasting", "🔀 Scenarios", "✅ Recommendations"])

# ------------------------------- OVERVIEW ----------------------------------
with tabs[0]:
    p = res["pop"]
    c = st.columns(4)
    with c[0]: card("Present students", f"{p['present']:,.0f}", f"of {p['total']:,.0f} enrolled")
    with c[1]: card("Overloaded systems", f"{res['overloaded']}", f"{res['at_risk']} at or above 90%")
    with c[2]: card("Bottleneck", res["bottleneck"], f"{res['util'][res['bottleneck']]:.0f}% utilisation",
                    res["util"][res["bottleneck"]])
    with c[3]: card("Congestion risk", f"{res['util']['Commuting']:.0f}", "commuting index (output)", res["util"]["Commuting"])
    cols = st.columns(6)
    for col, s in zip(cols, SYSTEMS):
        with col: card(s, f"{res['util'][s]:.0f}%", f"{res['util'][s] - res_ref['util'][s]:+.1f} pts vs normal", res["util"][s])
    util_bar(SYSTEMS, [res["util"][s] for s in SYSTEMS], "System utilisation (current scenario)")
    st.markdown("**Ripple summary vs the normal campus**")
    S = ge.node_states(inp, res, ref, res_ref)
    _, cascades = ge.ripple_report(ge.build_graph(ge.load_edges(data.get("dependencies"))), S)
    if cascades:
        for f_, sysl in cascades.items():
            st.write(f"- **{f_}** changed -> affects: {', '.join(sysl) if sysl else 'no system noticeably'}")
    else:
        st.info("Scenario equals the normal campus. Move a slider and watch the ripple spread.")

# ------------------------------- ELECTRICITY -------------------------------
with tabs[1]:
    e = res["elec"]
    c = st.columns(4)
    with c[0]: card("Daily demand", f"{e['daily']:,.0f} kWh", f"capacity {e['capacity_kwh']:,.0f}", e["daily_util"])
    with c[1]: card("Peak demand", f"{e['peak_kw']:,.0f} kW", f"capacity {e['capacity_kw']:,.0f}", e["peak_util"])
    with c[2]: card("Cooling load", f"{e['cooling']:,.0f} kWh", f"{temperature:.0f}C vs {A['cooling_threshold_c']:.0f}C threshold")
    with c[3]: card("Event load", f"{e['event']:,.0f} kWh", "visitors + stage/AV")
    l, r = st.columns(2)
    with l:
        fig = go.Figure(go.Bar(x=["Base", "Occupancy", "Class sessions", "Cooling", "Event"],
                               y=[e["base"], e["occupancy"], e["sessions"], e["cooling"], e["event"]],
                               marker_color=["#5B6B86", "#4DA3FF", "#4DA3FF", "#F1C40F", "#E67E22"]))
        fig.update_layout(title="Demand breakdown (kWh/day)", height=330)
        show(fig)
    with r:
        temps = np.arange(10, 46, 1.0)
        ys = [simulate(replace(inp, temperature_c=t), A)["elec"]["util"] for t in temps]
        fig = go.Figure(go.Scatter(x=temps, y=ys, mode="lines", line=dict(color="#4DA3FF", width=3)))
        fig.add_vline(x=temperature, line_dash="dot", line_color="#FFB020")
        fig.add_hline(y=90, line_dash="dot", line_color=RISK_COLORS["Critical"])
        fig.update_layout(title="Sensitivity: electricity utilisation vs temperature", height=330,
                          xaxis_title="Temperature (C)", yaxis_title="Utilisation %")
        show(fig)
    if em:
        f = fc.forecast_electricity_day(em, inp)
        st.markdown(f"**ML forecast for this scenario{tag}:** {f['daily_kwh']:,.0f} kWh/day "
                    f"(formula simulation: {e['daily']:,.0f} kWh)"
                    + (f", peak at {f['peak_hour']:02d}:00" if f["peak_hour"] is not None else ""))
        if f["hourly"] is not None:
            fig = px.line(f["hourly"], x="hour", y="forecast_kwh", markers=True, title="Forecast hourly profile")
            fig.add_hline(y=A["peak_capacity_kw"], line_dash="dot", line_color=RISK_COLORS["Overloaded"],
                          annotation_text="peak capacity")
            show(fig)
    if data.get("electricity_data") is not None:
        d = data["electricity_data"]
        daily = d.groupby(d["timestamp"].dt.normalize())["consumption_kwh"].sum().reset_index()
        show(px.line(daily, x="timestamp", y="consumption_kwh", title=f"Historical daily consumption{tag}"))

# ------------------------------- CLASSROOMS --------------------------------
with tabs[2]:
    k = res["classes"]
    c = st.columns(4)
    with c[0]: card("Peak utilisation", f"{k['peak_util']:.0f}%", f"period {k['peak_period']}", k["peak_util"])
    with c[1]: card("Avg class size", f"{k['class_size']:.0f}", f"room capacity {A['classroom_capacity']:.0f}")
    with c[2]: card("Overcrowded rooms", f"{k['overcrowded_rooms']:.0f}", f"of {A['total_classrooms']:.0f} rooms")
    with c[3]: card("Underutilised rooms", f"{k['underutilized_rooms']:.0f}", f"below {A['underutilized_threshold_pct']:.0f}% fill")
    util_bar([f"P{i + 1}" for i in range(len(k["util_by_period"]))], k["util_by_period"], "Classroom utilisation by period")
    st.caption(f"Peak periods: {k['peak_periods']} | overloaded periods: {k['over_periods'] or 'none'} | "
               f"underutilised periods: {k['under_periods'] or 'none'} | sessions without a room at peak: "
               f"{k['unscheduled_sessions']:.0f}")
    if data.get("classroom_data") is not None:
        h = fc.analyze_classroom_history(data["classroom_data"], A)
        st.markdown(f"**Historical classroom analysis{tag}**")
        st.info(h["message"])
        table(h["table"])
        show(px.bar(h["table"], x="period", y="avg_util_pct", title="Historical average utilisation by period"))

# --------------------------------- CANTEEN ---------------------------------
with tabs[3]:
    k = res["canteen"]
    c = st.columns(4)
    with c[0]: card("Meals required", f"{k['demand']:,.0f}", f"capacity {k['capacity']:,.0f}", k["util"])
    with c[1]: card("Event visitor meals", f"{k['event_meals']:,.0f}", "extra demand")
    with c[2]: card("Excess demand", f"{k['excess']:,.0f}", "meals above capacity")
    with c[3]: card("Potential food waste", f"{k['waste']:,.0f}", "prepared but not needed")
    fig = go.Figure()
    fig.add_bar(name="Students", x=["Demand"], y=[k["students_meals"]], marker_color="#4DA3FF")
    fig.add_bar(name="Event visitors", x=["Demand"], y=[k["event_meals"]], marker_color="#E67E22")
    fig.add_bar(name="Capacity", x=["Capacity"], y=[k["capacity"]], marker_color="#5B6B86")
    fig.add_bar(name="Prepared", x=["Prepared"], y=[k["prepared"]], marker_color="#2ECC71")
    fig.update_layout(barmode="stack", title="Meal demand vs capacity vs preparation", height=360)
    show(fig)
    if cm_:
        st.markdown(f"**ML forecast for this scenario{tag}:** {fc.forecast_canteen_day(cm_, inp):,.0f} meals "
                    f"(formula simulation: {k['demand']:,.0f})")
    if data.get("canteen_data") is not None:
        d = data["canteen_data"]
        cols = st.columns(2)
        with cols[0]:
            show(px.line(d, x="date", y="meals_served", title=f"Historical daily meals{tag}"))
        if "attendance_rate" in d:
            with cols[1]:
                show(px.scatter(d, x="attendance_rate", y="meals_served", color=d["event_flag"].map({0: "normal", 1: "event"}),
                                title="Meals vs attendance"))

# ------------------------------- COMMUTING ---------------------------------
with tabs[4]:
    m = res["commute"]
    st.caption("Commuting is an OUTPUT: derived from present students, rain, timetable concentration and events.")
    c = st.columns(4)
    with c[0]: card("Bus utilisation", f"{m['bus_util']:.0f}%", f"{m['bus_demand_peak']:,.0f} riders vs {m['bus_capacity_peak']:,.0f} seats", m["bus_util"])
    with c[1]: card("Parking utilisation", f"{m['parking_util']:.0f}%", f"{m['parking_demand']:,.0f} of {m['parking_capacity']:,.0f}", m["parking_util"])
    with c[2]: card("Arrival pressure", f"{m['road_util']:.0f}%", f"peak {m['peak_arrival']} ({m['peak_share'] * 100:.0f}% arrive)", m["road_util"])
    with c[3]: card("Congestion risk", f"{m['congestion']:.0f}", f"CO2 est. {m['emissions_kg']:,.0f} kg/day", m["congestion"])
    l, r = st.columns(2)
    with l:
        ref_sh = res_ref["commute"]["shares"]
        fig = go.Figure()
        fig.add_bar(name="Normal campus", x=list(ref_sh), y=[v * 100 for v in ref_sh.values()], marker_color="#5B6B86")
        fig.add_bar(name="Current scenario", x=list(m["shares"]), y=[v * 100 for v in m["shares"].values()], marker_color="#4DA3FF")
        fig.update_layout(barmode="group", title="Commuting mode share (%)", height=340)
        show(fig)
    with r:
        rows = []
        for label, rn in (("Dry day", 0.0), ("Light rain (5 mm)", 5.0), ("Heavy rain (25 mm)", 25.0)):
            rr = simulate(replace(inp, rainfall_mm=rn), A)["commute"]
            rows.append(dict(Scenario=label, Bus=rr["bus_util"], Parking=rr["parking_util"], Congestion=rr["congestion"]))
        fig = px.bar(pd.DataFrame(rows), x="Scenario", y=["Bus", "Parking", "Congestion"], barmode="group",
                     title="Commuting scenarios: effect of rain (same other inputs)")
        fig.update_layout(height=340, yaxis_title="Index / utilisation %")
        show(fig)
    st.caption(f"Estimated commuters: {m['commuters']:,.0f} | bus riders: {m['bus_riders']:,.0f} | cars: {m['cars']:,.0f} | "
               f"two-wheelers: {m['two_wheelers']:,.0f} | visitor vehicles: {m['visitor_vehicles']:,.0f} | rain shift factor: {m['rain_factor']:.2f}")

# --------------------------------- WATER -----------------------------------
with tabs[5]:
    w = res["water"]
    c = st.columns(3)
    with c[0]: card("Daily demand", f"{w['demand']:,.0f} L", f"capacity {w['capacity']:,.0f} L", w["util"])
    with c[1]: card("Infrastructure pressure", risk_level(w["util"]), f"{w['util']:.0f}% of supply", w["util"])
    with c[2]: card("Heat multiplier", f"x{w['heat_mult']:.2f}", f"{temperature:.0f}C")
    fig = go.Figure(go.Bar(x=["Present students", "Hostel residents", "Event visitors", "Canteen/meals"],
                           y=[w["students"], w["hostel"], w["event"], w["meals"]],
                           marker_color=["#4DA3FF", "#5B6B86", "#E67E22", "#2ECC71"]))
    fig.update_layout(title="Water demand sources (L/day, before heat & saving factors)", height=340)
    show(fig)

# ----------------------------- BUTTERFLY GRAPH -----------------------------
with tabs[6]:
    edges = ge.load_edges(data.get("dependencies"))
    G = ge.build_graph(edges)
    S = ge.node_states(inp, res, ref, res_ref)
    st.caption("Blue = changed input | risk colours = utilisation | amber arrows = ripple paths active vs the normal campus. "
               "Hover nodes and arrows for explanations.")
    show(ge.make_figure(G, S))
    lines, cascades = ge.ripple_report(G, S)
    l, r = st.columns(2)
    with l:
        st.markdown("**Active impacts**")
        for ln in lines or ["No active ripples - change a control in the sidebar."]:
            st.write("- " + ln)
    with r:
        node = st.selectbox("Inspect a node", list(G.nodes))
        ins, outs = ge.explain_node(G, S, node)
        st.write(f"**{node}** - {S[node]['text']} ({S[node]['delta']})")
        st.write("Driven by: " + ("; ".join(ins) if ins else "an independent input"))
        st.write("Affects: " + ("; ".join(outs) if outs else "nothing downstream"))
    with st.expander("Dependency verification (perturbs each factor and re-runs the simulation)"):
        table(ge.verify_edges(inp, A, edges))

# -------------------------------- FORECASTING ------------------------------
with tabs[7]:
    st.caption("Random Forest, time-ordered 80/20 split, compared with a seasonal-naive baseline."
               + (" **Datasets flagged as synthetic: results are demonstrations, not real campus accuracy.**" if synthetic else ""))
    for b, title in ((em, "Electricity demand model"), (cm_, "Canteen demand model")):
        st.subheader(title)
        if not b:
            st.info("No usable dataset for this model - add the CSV (see README).")
            continue
        mt = b["metrics"]
        c = st.columns(4)
        c[0].metric("Model MAE", f"{mt['model_mae']:,.1f} {b['unit']}")
        c[1].metric("Model RMSE", f"{mt['model_rmse']:,.1f}")
        c[2].metric("Baseline MAE", f"{mt['base_mae']:,.1f}")
        c[3].metric("MAE vs baseline", f"{mt['mae_improvement_pct']:+.0f}%")
        t = b["test"].melt("time", var_name="series", value_name="value")
        show(px.line(t, x="time", y="value", color="series", title=f"Actual vs predicted (test set, {b['n_test']} rows){tag}"))
        l, r = st.columns(2)
        with l:
            show(px.bar(b["importance"], orientation="h", title="Feature importance"))
        with r:
            better = mt["mae_improvement_pct"] > 0
            st.markdown(
                f"- Trained on the first {b['n_train']} rows, tested on the last {b['n_test']} (no shuffling).\n"
                f"- MAE is the average miss ({mt['model_mae']:,.1f} {b['unit']}, {mt['model_mae'] / max(mt['mean_target'], 1e-9) * 100:.1f}% "
                f"of the mean).\n- The model {'beats' if better else 'does NOT beat'} the seasonal-naive baseline "
                f"by {abs(mt['mae_improvement_pct']):.0f}% on MAE.\n- Features: {', '.join(b['features'])}.")

# --------------------------------- SCENARIOS --------------------------------
with tabs[8]:
    st.subheader("Parallel campus scenarios")
    with st.expander("Scenario B (stress) and C (interventions) settings", expanded=False):
        c = st.columns(6)
        stress = dict(
            attendance_delta=c[0].number_input("+ Attendance (pts)", 0.0, 50.0, sa.DEFAULT_STRESS["attendance_delta"], key="s_att"),
            temp_delta=c[1].number_input("+ Temperature (C)", 0.0, 25.0, sa.DEFAULT_STRESS["temp_delta"], key="s_t"),
            rainfall_mm=c[2].number_input("Rainfall (mm)", 0.0, 80.0, sa.DEFAULT_STRESS["rainfall_mm"], key="s_r"),
            event_visitors=c[3].number_input("Event visitors", 0, 100000, sa.DEFAULT_STRESS["event_visitors"], 100, key="s_v"),
            event_intensity=c[4].selectbox("Event intensity", [1, 2, 3], 2, key="s_i"),
            conc_delta=c[5].number_input("+ Peak concentration (pts)", 0.0, 50.0, sa.DEFAULT_STRESS["conc_delta"], key="s_c"))
        c = st.columns(5)
        ivs = dict(
            stagger_pct=c[0].number_input("Stagger (%)", 0.0, 100.0, sa.DEFAULT_INTERVENTIONS["stagger_pct"], key="i_s"),
            extra_buses=int(c[1].number_input("Extra buses", 0, 60, sa.DEFAULT_INTERVENTIONS["extra_buses"], key="i_b")),
            allocation_efficiency_pct=c[2].number_input("Allocation eff. (%)", 50.0, 100.0, sa.DEFAULT_INTERVENTIONS["allocation_efficiency_pct"], key="i_a"),
            canteen_capacity_boost_pct=c[3].number_input("Canteen boost (%)", 0.0, 200.0, sa.DEFAULT_INTERVENTIONS["canteen_capacity_boost_pct"], key="i_c"),
            energy_mgmt_pct=c[4].number_input("Energy mgmt (%)", 0.0, 60.0, sa.DEFAULT_INTERVENTIONS["energy_mgmt_pct"], key="i_e"))
    scen = sa.build_scenarios(ref, stress, ivs)
    tbl, sres = sa.compare(scen, A, em, cm_, fc.forecast_electricity_day, fc.forecast_canteen_day)
    fig = go.Figure()
    for name, color in zip(scen, ("#5B6B86", "#E67E22", "#2ECC71")):
        fig.add_bar(name=name, x=SYSTEMS, y=[sres[name]["util"][s] for s in SYSTEMS], marker_color=color)
    fig.add_hline(y=100, line_dash="dot", line_color=RISK_COLORS["Overloaded"])
    fig.update_layout(barmode="group", title="Utilisation by system and scenario (%)", height=380)
    show(fig)
    st.dataframe(tbl.style.format("{:,.1f}"))
    ob, oc = sres[list(scen)[1]]["overloaded"], sres[list(scen)[2]]["overloaded"]
    if oc < ob:
        st.success(f"Overloaded systems: Scenario B = {ob}, Scenario C = {oc}.")
    else:
        st.info(f"Overloaded systems: B = {ob}, C = {oc}. Strengthen the interventions above.")
    df_i, bott, best, base_b = sa.intervention_impacts(scen[list(scen)[1]], A, ivs)
    st.markdown(f"**Bottleneck in Scenario B: {bott} ({base_b['util'][bott]:.0f}%).** Each intervention applied alone:")
    table(df_i)
    st.success(f"Best single fix for the bottleneck: **{best['Intervention']}** ({best['Change (pts)']:+.1f} pts on {bott}).")

# ------------------------------ RECOMMENDATIONS -----------------------------
with tabs[9]:
    recs = rec.generate(inp, res, A, ref)
    if not recs:
        st.success("No system is at or above 90% utilisation in this scenario. Raise attendance, temperature or add an event to stress the campus.")
    for r in recs:
        icon = {"Overloaded": "🔴", "Critical": "🟠"}.get(r["level"], "🟡")
        with st.expander(f"{icon} {r['system']} - {r['level']} ({r['before']:.0f}%)", expanded=True):
            st.markdown(f"**What is wrong:** {r['what']}")
            st.markdown("**Contributing factors:** " + (", ".join(f"{n} (+{d:.0f} pts)" for n, d in r["drivers"]) or
                                                         "capacity is tight even under normal conditions"))
            st.markdown(f"**Suggested action:** {r['action']}")
            if r["after"] is not None:
                st.markdown(f"**Simulated improvement:** {r['system']} {r['before']:.0f}% -> **{r['after']:.0f}%** "
                            f"({'target reached' if r['resolved'] else 'improved but still above 85% - combine measures'}); "
                            f"overloaded systems after: {r['overloaded_after']}.")
                if r["side_effects"]:
                    st.warning("Side effects: " + "; ".join(r["side_effects"]))

st.markdown('<p class="note">All thresholds, capacities and coefficients are assumptions (editable above). '
            'Values from synthetic datasets are not measured campus statistics.</p>', unsafe_allow_html=True)
