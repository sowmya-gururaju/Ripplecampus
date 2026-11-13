"""
simulation.py - MIRROR simulation engine.

Pure functions, no Streamlit. Every number is computed from the user's inputs
(ScenarioInputs) and the configurable assumptions dict (A). Nothing is hardcoded.

Commuting is an OUTPUT: it is derived from present students, weather, timetable
and events using the commuting assumptions (mode shares, rain shift, capacities).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace, asdict  # noqa: F401 (re-exported)

EPS = 1e-9
SYSTEMS = ["Electricity", "Classrooms", "Canteen", "Water", "Parking", "Transport"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
RISK_COLORS = {"Healthy": "#2ECC71", "Warning": "#F1C40F", "Critical": "#E67E22", "Overloaded": "#E74C3C"}

# --------------------------------------------------------------------------
# Assumptions: key -> (default value, description). All editable in the UI and
# overridable through data/campus_baseline.csv (columns: key,value).
# DEFAULTS ARE PLACEHOLDERS - replace with your campus' real numbers.
# --------------------------------------------------------------------------
DEFAULT_ASSUMPTIONS = {
    # Baseline ("Normal campus") scenario
    "total_students": (5000, "Enrolled students before intake growth"),
    "typical_attendance_pct": (75, "Typical attendance on a normal day (%)"),
    "typical_temperature_c": (30, "Typical daily temperature (C)"),
    "typical_simultaneous_classes": (40, "Typical classes running per period"),
    "typical_peak_concentration_pct": (50, "Typical % of daily classes held in peak periods"),
    # Population
    "hostel_rate": (0.30, "Share of students living on campus (they do not commute)"),
    "canteen_participation": (0.65, "Share of present students using the canteen"),
    "meals_per_participant": (1.5, "Meals per participating student per day"),
    # Classrooms
    "total_classrooms": (80, "Number of classrooms"),
    "classroom_capacity": (60, "Seats per classroom"),
    "periods_per_day": (8, "Teaching periods per day"),
    "peak_periods": (3, "Number of consecutive peak periods"),
    "peak_period_start": (2, "First peak period (1-based)"),
    "periods_attended_per_student": (4, "Periods a present student attends per day"),
    "allocation_efficiency_pct": (80, "How well classes are matched to rooms (%)"),
    "class_size_cv": (0.30, "Class-size variability (std/mean) at 80% allocation efficiency"),
    "underutilized_threshold_pct": (40, "A room below this fill % is underutilised"),
    # Electricity
    "electricity_capacity_kwh": (48000, "Daily electricity capacity (kWh)"),
    "peak_capacity_kw": (5000, "Peak power capacity (kW)"),
    "base_load_kwh": (14000, "Base load with an empty campus (kWh/day)"),
    "kwh_per_present_student": (2.5, "Occupancy load per present student (kWh/day)"),
    "kwh_per_class_session": (12, "Load per class session - lights, AC, projector (kWh)"),
    "cooling_threshold_c": (24, "Cooling starts above this temperature (C)"),
    "cooling_kwh_per_degree": (900, "Cooling kWh per C above threshold at full occupancy"),
    "operating_hours": (12, "Hours per day the campus is active"),
    "peak_base_uplift": (0.15, "Peak-to-average uplift with perfectly spread timetable"),
    "peak_conc_uplift": (0.90, "Extra peak uplift when ALL classes are in peak periods"),
    "event_kwh_per_visitor": (0.8, "Event electricity per visitor (kWh)"),
    "event_fixed_kwh": (1500, "Fixed event electricity per intensity level (stage, lighting, AV)"),
    # Canteen
    "canteen_capacity_meals": (6000, "Meals the canteen can serve per day"),
    "canteen_planned_meals": (0, "Meals normally prepared (0 = auto from the normal day)"),
    "canteen_prep_buffer": (0.10, "Extra preparation buffer used when planned meals is auto"),
    "visitor_meals": (1.0, "Meals per event visitor"),
    "canteen_heat_uplift_per_c": (0.01, "Extra canteen demand per C above 30C"),
    # Water
    "water_capacity_l": (350000, "Daily water supply capacity (litres)"),
    "water_lpd_per_present_student": (25, "Litres per present student per day"),
    "water_lpd_per_hostel_resident": (90, "Litres per hostel resident per day"),
    "water_lpd_per_visitor": (12, "Litres per event visitor"),
    "water_per_meal_l": (6, "Litres of water per meal prepared"),
    "water_heat_uplift_per_c": (0.015, "Extra water demand per C above 28C"),
    # Events
    "event_intensity_slope": (0.4, "Per-person demand multiplier increase per intensity level"),
    # Commuting (baseline shares of commuting students; normalised automatically)
    "share_bus": (0.30, "Baseline share using buses"),
    "share_two_wheeler": (0.30, "Baseline share using two-wheelers"),
    "share_car": (0.15, "Baseline share using cars"),
    "share_walk_cycle": (0.25, "Baseline share walking/cycling"),
    "rain_saturation_mm": (20, "Rainfall at which the full rain mode shift applies (mm)"),
    "rain_2w_to_bus": (0.25, "Fraction of two-wheeler users moving to bus in full rain"),
    "rain_2w_to_car": (0.15, "Fraction of two-wheeler users moving to car in full rain"),
    "rain_walk_to_bus": (0.15, "Fraction of walkers/cyclists moving to bus in full rain"),
    "rain_walk_to_car": (0.10, "Fraction of walkers/cyclists moving to car in full rain"),
    "buses": (6, "Campus buses available"),
    "bus_capacity": (50, "Passengers per bus"),
    "trips_per_bus_peak_window": (2, "Trips one bus can make during the peak arrival window"),
    "parking_capacity": (850, "Parking capacity in car-equivalent spaces"),
    "car_occupancy": (1.8, "People per car"),
    "two_wheeler_occupancy": (1.2, "People per two-wheeler"),
    "two_wheeler_space_factor": (0.25, "Car-equivalent parking space of one two-wheeler"),
    "road_throughput_veh_per_hr": (400, "Gate/road throughput (passenger-car units per hour)"),
    "peak_window_hours": (1.5, "Length of the peak arrival window (hours)"),
    "arrival_base_share": (0.30, "Share of arrivals in peak window with a fully spread timetable"),
    "arrival_conc_weight": (0.5, "How strongly timetable concentration raises the peak arrival share"),
    "first_class_hour": (9.0, "Hour of the first class (24h clock)"),
    "visitor_bus_share": (0.20, "Share of event visitors arriving by bus/shuttle"),
    "visitor_vehicle_share": (0.50, "Share of event visitors arriving by private vehicle"),
    "visitor_vehicle_occupancy": (2.5, "Visitors per vehicle"),
    "pcu_two_wheeler": (0.5, "Passenger-car units of one two-wheeler"),
    "pcu_bus": (2.5, "Passenger-car units of one bus"),
    "avg_commute_km": (8, "Average one-way commute distance (km)"),
    "ef_car_kg_per_km": (0.17, "CO2 per car-km (kg)"),
    "ef_two_wheeler_kg_per_km": (0.07, "CO2 per two-wheeler-km (kg)"),
    "ef_bus_kg_per_km": (0.85, "CO2 per bus-km (kg)"),
}
DEFAULT_VALUES = {k: v[0] for k, v in DEFAULT_ASSUMPTIONS.items()}


@dataclass
class ScenarioInputs:
    # Independent input factors (commuting is NOT an input - it is an output)
    enrolled_students: float = 5000
    intake_growth_pct: float = 0.0
    attendance_rate: float = 75.0          # %
    temperature_c: float = 30.0
    rainfall_mm: float = 0.0
    simultaneous_classes: float = 40
    peak_concentration_pct: float = 50.0   # % of classes in peak periods
    event_day: bool = False
    event_visitors: float = 0
    event_intensity: int = 2               # 1 low, 2 medium, 3 high
    day_of_week: int = 2                   # 0=Mon (used by the ML forecasts)
    # Intervention levers (all neutral by default)
    stagger_pct: float = 0.0               # % of excess peak load moved off-peak
    allocation_efficiency_pct: float = 80.0
    extra_buses: int = 0
    canteen_capacity_boost_pct: float = 0.0
    energy_mgmt_pct: float = 0.0           # % cooling reduction (+ half as much peak shaving)
    modal_shift_pct: float = 0.0           # % of private-vehicle commuters moved to buses
    water_saving_pct: float = 0.0


# ----------------------------- helpers ------------------------------------
def _f(x, default=0.0, lo=None, hi=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = float(default)
    if math.isnan(v) or math.isinf(v):
        v = float(default)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def pct_util(demand, capacity):
    """demand / capacity * 100, safe against zero/negative capacity."""
    demand = max(0.0, demand)
    if capacity <= EPS:
        return 0.0 if demand <= EPS else 999.0
    return min(999.0, demand / capacity * 100.0)


def _cdf(x, mean, sd):
    if sd <= EPS:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1 + math.erf((x - mean) / (sd * math.sqrt(2))))


def risk_level(u):
    if u < 70:
        return "Healthy"
    if u < 90:
        return "Warning"
    if u <= 100:
        return "Critical"
    return "Overloaded"


def risk_color(u):
    return RISK_COLORS[risk_level(u)]


def _fmt_time(h):
    h = h % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


# ----------------------------- simulation ---------------------------------
def simulate(inp: ScenarioInputs, A: dict | None = None) -> dict:
    A = {**DEFAULT_VALUES, **(A or {})}

    def a(k):
        return _f(A.get(k), DEFAULT_VALUES[k], lo=0.0)

    # ---- population (no double counting: everything derives from `present`)
    enrolled = _f(inp.enrolled_students, 0, lo=0)
    growth = _f(inp.intake_growth_pct, 0, lo=-100, hi=500)
    total = enrolled * (1 + growth / 100)
    new_students = total - enrolled
    att = _f(inp.attendance_rate, 0, 0, 100) / 100
    present = total * att
    hostel = total * min(1.0, a("hostel_rate"))
    commuters = present * (1 - min(1.0, a("hostel_rate")))  # hostel students do not commute
    ev = bool(inp.event_day)
    visitors = _f(inp.event_visitors, 0, lo=0) if ev else 0.0
    intensity = int(_f(inp.event_intensity, 2, 1, 3))
    imult = (1 + a("event_intensity_slope") * (intensity - 1)) if ev else 0.0
    temp = _f(inp.temperature_c, 30, -20, 60)
    rain = _f(inp.rainfall_mm, 0, lo=0)

    # ---- timetable shape
    periods = max(1, int(round(a("periods_per_day"))))
    peak_n = int(min(max(1, round(a("peak_periods"))), periods))
    start = int(min(max(1, round(a("peak_period_start"))), periods - peak_n + 1))
    c_uniform = peak_n / periods
    c0 = _f(inp.peak_concentration_pct, 50, 0, 100) / 100
    stagger = _f(inp.stagger_pct, 0, 0, 100) / 100
    c_eff = c0 - max(0.0, c0 - c_uniform) * stagger
    sim_classes = _f(inp.simultaneous_classes, 0, lo=0)
    n_sessions = sim_classes * periods
    if peak_n >= periods:
        sessions = [n_sessions / periods] * periods
    else:
        pk = n_sessions * c_eff / peak_n
        off = n_sessions * (1 - c_eff) / (periods - peak_n)
        sessions = [pk if start <= p + 1 < start + peak_n else off for p in range(periods)]
    peak_periods_list = [p + 1 for p in range(periods) if start <= p + 1 < start + peak_n]

    # ---- classrooms
    rooms = a("total_classrooms")
    cap = a("classroom_capacity")
    alloc = _f(inp.allocation_efficiency_pct, 80, 30, 100) / 100
    class_size = present * a("periods_attended_per_student") / n_sessions if n_sessions > EPS else 0.0
    students_p = [s * class_size for s in sessions]
    seat_cap = rooms * cap * alloc
    util_p = [max(pct_util(sp, seat_cap), pct_util(s, rooms)) for sp, s in zip(students_p, sessions)]
    peak_idx = max(range(periods), key=lambda i: util_p[i])
    sessions_peak = sessions[peak_idx]
    rooms_in_use = min(sessions_peak, rooms)
    cv = max(0.05, a("class_size_cv") * (1.2 - alloc) / 0.4)
    sd = cv * class_size
    over_rooms = rooms_in_use * (1 - _cdf(cap, class_size, sd))
    under_thr = a("underutilized_threshold_pct") / 100 * cap
    under_rooms = (rooms - rooms_in_use) + rooms_in_use * _cdf(under_thr, class_size, sd)
    class_res = dict(
        sessions_by_period=sessions, students_by_period=students_p, util_by_period=util_p,
        class_size=class_size, peak_period=peak_idx + 1, peak_util=util_p[peak_idx],
        avg_util=sum(util_p) / periods, rooms_in_use=rooms_in_use,
        overcrowded_rooms=min(rooms, over_rooms), underutilized_rooms=min(rooms, under_rooms),
        unscheduled_sessions=max(0.0, sessions_peak - rooms), c_eff=c_eff, seat_capacity=seat_cap,
        peak_periods=peak_periods_list,
        over_periods=[i + 1 for i, u in enumerate(util_p) if u > 100],
        under_periods=[i + 1 for i, u in enumerate(util_p) if u < a("underutilized_threshold_pct")],
        util=util_p[peak_idx],
    )

    # ---- canteen
    heat_c = 1 + a("canteen_heat_uplift_per_c") * max(0.0, temp - 30)
    meals_students = present * min(1.0, a("canteen_participation")) * a("meals_per_participant") * heat_c
    meals_event = visitors * a("visitor_meals") * imult
    demand_meals = meals_students + meals_event
    base_cap = a("canteen_capacity_meals")
    cap_meals = base_cap * (1 + _f(inp.canteen_capacity_boost_pct, 0, 0, 500) / 100)
    planned = a("canteen_planned_meals") or 0.8 * base_cap
    prepared = min(planned * (1 + _f(inp.canteen_capacity_boost_pct, 0, 0, 500) / 100), cap_meals)
    canteen_res = dict(
        students_meals=meals_students, event_meals=meals_event, demand=demand_meals,
        capacity=cap_meals, served=min(demand_meals, cap_meals), excess=max(0.0, demand_meals - cap_meals),
        prepared=prepared, waste=max(0.0, prepared - demand_meals), util=pct_util(demand_meals, cap_meals),
    )

    # ---- water
    heat_w = 1 + a("water_heat_uplift_per_c") * max(0.0, temp - 28)
    saving = 1 - _f(inp.water_saving_pct, 0, 0, 90) / 100
    w_students = present * a("water_lpd_per_present_student")
    w_hostel = hostel * a("water_lpd_per_hostel_resident")
    w_event = visitors * a("water_lpd_per_visitor") * imult
    w_meals = demand_meals * a("water_per_meal_l")
    w_total = (w_students + w_hostel + w_event + w_meals) * heat_w * saving
    water_res = dict(students=w_students, hostel=w_hostel, event=w_event, meals=w_meals,
                     heat_mult=heat_w, demand=w_total, capacity=a("water_capacity_l"),
                     util=pct_util(w_total, a("water_capacity_l")))

    # ---- electricity
    mgmt = _f(inp.energy_mgmt_pct, 0, 0, 100) / 100
    e_base = a("base_load_kwh")
    e_occ = present * a("kwh_per_present_student")
    e_sess = n_sessions * a("kwh_per_class_session")
    e_cool = max(0.0, temp - a("cooling_threshold_c")) * a("cooling_kwh_per_degree") * (0.4 + 0.6 * att) * (1 - mgmt)
    e_event = (visitors * a("event_kwh_per_visitor") * imult + a("event_fixed_kwh") * intensity) if ev else 0.0
    e_daily = e_base + e_occ + e_sess + e_cool + e_event
    avg_kw = e_daily / max(1.0, a("operating_hours"))
    excess = max(0.0, c_eff - c_uniform) / max(EPS, 1 - c_uniform)
    peak_factor = 1 + a("peak_base_uplift") + a("peak_conc_uplift") * excess
    peak_kw = avg_kw * peak_factor * (1 - 0.5 * mgmt)
    d_util = pct_util(e_daily, a("electricity_capacity_kwh"))
    p_util = pct_util(peak_kw, a("peak_capacity_kw"))
    elec_res = dict(base=e_base, occupancy=e_occ, sessions=e_sess, cooling=e_cool, event=e_event,
                    daily=e_daily, avg_kw=avg_kw, peak_kw=peak_kw, peak_factor=peak_factor,
                    capacity_kwh=a("electricity_capacity_kwh"), capacity_kw=a("peak_capacity_kw"),
                    daily_util=d_util, peak_util=p_util, util=max(d_util, p_util))

    # ---- commuting (OUTPUT)
    raw = [a("share_bus"), a("share_two_wheeler"), a("share_car"), a("share_walk_cycle")]
    if sum(raw) <= EPS:
        raw = [DEFAULT_VALUES[k] for k in ("share_bus", "share_two_wheeler", "share_car", "share_walk_cycle")]
    tot_s = sum(raw)
    s_bus, s_2w, s_car, s_walk = [x / tot_s for x in raw]
    rf = min(1.0, rain / max(a("rain_saturation_mm"), EPS))
    t2b = s_2w * rf * min(1.0, a("rain_2w_to_bus"))
    t2c = s_2w * rf * min(1.0, a("rain_2w_to_car"))
    w2b = s_walk * rf * min(1.0, a("rain_walk_to_bus"))
    w2c = s_walk * rf * min(1.0, a("rain_walk_to_car"))
    bus_s, car_s = s_bus + t2b + w2b, s_car + t2c + w2c
    two_s, walk_s = s_2w - t2b - t2c, s_walk - w2b - w2c
    m = _f(inp.modal_shift_pct, 0, 0, 100) / 100
    moved = (car_s + two_s) * m
    bus_s += moved
    car_s *= (1 - m)
    two_s *= (1 - m)
    shares = {"Bus": bus_s, "Two-wheeler": two_s, "Car": car_s, "Walk/Cycle": walk_s}

    bus_riders = commuters * bus_s + visitors * a("visitor_bus_share")
    cars = commuters * car_s / max(a("car_occupancy"), 0.5)
    two_w = commuters * two_s / max(a("two_wheeler_occupancy"), 0.5)
    visitor_veh = visitors * a("visitor_vehicle_share") / max(a("visitor_vehicle_occupancy"), 0.5)
    peak_share = min(0.9, max(0.2, a("arrival_base_share") + a("arrival_conc_weight") * c_eff))
    buses_total = max(0.0, a("buses") + _f(inp.extra_buses, 0, lo=0))
    bus_cap_peak = buses_total * a("bus_capacity") * a("trips_per_bus_peak_window")
    bus_demand_peak = bus_riders * peak_share
    bus_util = pct_util(bus_demand_peak, bus_cap_peak)
    parking_demand = cars + two_w * a("two_wheeler_space_factor") + visitor_veh
    park_util = pct_util(parking_demand, a("parking_capacity"))
    bus_trips_peak = bus_demand_peak / max(a("bus_capacity"), 1.0)
    pcu_peak = (cars + two_w * a("pcu_two_wheeler") + visitor_veh) * peak_share + bus_trips_peak * a("pcu_bus")
    road_util = pct_util(pcu_peak, a("road_throughput_veh_per_hr") * max(a("peak_window_hours"), 0.1))
    congestion = min(150.0, 0.5 * road_util + 0.25 * park_util + 0.25 * min(bus_util, 150))
    buses_running = min(buses_total, math.ceil(bus_riders / max(a("bus_capacity") * a("trips_per_bus_peak_window"), 1.0)))
    km = a("avg_commute_km") * 2
    emissions = cars * km * a("ef_car_kg_per_km") + two_w * km * a("ef_two_wheeler_kg_per_km") \
        + buses_running * km * a("ef_bus_kg_per_km")
    pk_end = a("first_class_hour")
    commute_res = dict(
        shares=shares, commuters=commuters, bus_riders=bus_riders, cars=cars, two_wheelers=two_w,
        visitor_vehicles=visitor_veh, peak_share=peak_share, bus_demand_peak=bus_demand_peak,
        bus_capacity_peak=bus_cap_peak, buses_total=buses_total, bus_util=bus_util,
        parking_demand=parking_demand, parking_capacity=a("parking_capacity"), parking_util=park_util,
        road_util=road_util, congestion=congestion, emissions_kg=emissions,
        peak_arrival=f"{_fmt_time(pk_end - a('peak_window_hours'))} - {_fmt_time(pk_end)}",
        rain_factor=rf,
    )

    util = {"Electricity": elec_res["util"], "Classrooms": class_res["util"], "Canteen": canteen_res["util"],
            "Water": water_res["util"], "Parking": park_util, "Transport": bus_util, "Commuting": congestion}
    sys_vals = [util[s] for s in SYSTEMS]
    return dict(
        pop=dict(total=total, new_students=new_students, present=present, hostel=hostel,
                 commuters=commuters, visitors=visitors, event_mult=imult),
        elec=elec_res, classes=class_res, canteen=canteen_res, water=water_res, commute=commute_res,
        util=util,
        overloaded=sum(1 for u in sys_vals if u > 100),
        at_risk=sum(1 for u in sys_vals if u >= 90),
        bottleneck=max(SYSTEMS, key=lambda s: util[s]),
    )
