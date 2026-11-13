"""
data_loader.py - loads, cleans and maps CSV datasets to what MIRROR needs.

Plug-in model: drop CSV files with the standard names into data/ (or upload them in the
sidebar). Column names are normalised (lower-case, underscores) and common aliases are
mapped automatically (see ALIASES). Missing datasets are fine - the matching feature just
falls back to the configurable assumptions.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from simulation import DEFAULT_VALUES, ScenarioInputs

DATA_DIR = Path(__file__).parent / "data"
DATASETS = ["campus_baseline", "electricity_data", "canteen_data", "classroom_data",
            "commuting_data", "weather_data", "campus_events", "dependencies"]

# canonical column -> accepted alternative names (after normalising to snake_case)
ALIASES = {
    "electricity_data": {
        "timestamp": ["datetime", "date_time", "time", "date", "ts", "timestamp"],
        "consumption_kwh": ["kwh", "energy_kwh", "electricity_kwh", "usage_kwh", "consumption",
                            "energy_consumption", "meter_reading", "load_kwh", "electricity_consumption"],
        "temperature_c": ["temperature", "temp", "temp_c", "avg_temp", "air_temperature", "airtemperature"],
        "attendance_rate": ["attendance", "attendance_pct", "attendance_percent", "attendance_percentage"],
        "event_flag": ["is_event", "event", "event_day", "event_indicator"],
        "occupancy": ["occupancy_rate", "occupancy_pct"],
    },
    "canteen_data": {
        "date": ["day", "datetime", "timestamp"],
        "meals_served": ["meals", "meal_demand", "daily_meals", "demand", "meals_sold", "footfall", "sales_count"],
        "attendance_rate": ["attendance", "attendance_pct", "attendance_percent"],
        "event_flag": ["is_event", "event", "event_day", "event_indicator"],
        "total_students": ["students", "enrolled", "enrolled_students", "student_population"],
        "temperature_c": ["temperature", "temp", "temp_c", "avg_temp"],
    },
    "classroom_data": {
        "date": ["day", "datetime"],
        "period": ["slot", "time_slot", "hour", "class_period"],
        "room_id": ["room", "classroom", "classroom_id", "room_no"],
        "occupancy": ["students_present", "students", "occupied", "headcount", "enrolled", "attendance"],
        "capacity": ["room_capacity", "seats", "seat_capacity"],
    },
    "commuting_data": {
        "date": ["day", "datetime"],
        "bus": ["buses", "bus_count", "bus_users"],
        "two_wheeler": ["two_wheelers", "bike", "bikes", "motorcycle", "two_wheeler_count", "twowheeler"],
        "car": ["cars", "car_count", "four_wheeler"],
        "walk_cycle": ["walk", "walking", "walk_or_cycle", "cycle", "walk_cycle_count", "walking_cycling"],
        "rainfall_mm": ["rain_mm", "rainfall", "rain", "precipitation"],
    },
    "weather_data": {
        "date": ["day", "datetime", "timestamp", "time"],
        "temperature_c": ["temperature", "temp", "temp_c", "avg_temp", "mean_temp"],
        "rainfall_mm": ["rain_mm", "rainfall", "rain", "precipitation", "precip_mm"],
    },
    "campus_events": {
        "date": ["day", "event_date", "datetime"],
        "event_name": ["event", "name", "title"],
        "visitors": ["expected_visitors", "attendance", "footfall", "participants"],
        "intensity": ["event_intensity", "level", "scale"],
    },
    "campus_baseline": {"key": ["parameter", "name", "setting"], "value": ["val", "amount"]},
    "dependencies": {"source": ["from", "src", "cause"], "target": ["to", "tgt", "effect"],
                     "description": ["explanation", "impact", "desc"]},
}
REQUIRED = {
    "electricity_data": ["timestamp", "consumption_kwh"],
    "canteen_data": ["date", "meals_served"],
    "classroom_data": ["period", "occupancy"],
    "commuting_data": ["bus", "two_wheeler", "car", "walk_cycle"],
    "weather_data": ["date", "temperature_c"],
    "campus_events": ["date"],
    "campus_baseline": ["key", "value"],
    "dependencies": ["source", "target"],
}


# ------------------------------ cleaning ----------------------------------
def _snake(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def _canonical(df, aliases):
    df = df.copy()
    df.columns = [_snake(c) for c in df.columns]
    rename = {}
    for canon, alts in aliases.items():
        if canon in df.columns:
            continue
        for alt in alts:
            if alt in df.columns and alt not in rename:
                rename[alt] = canon
                break
    return df.rename(columns=rename)


def _flag(s):
    num = pd.to_numeric(s, errors="coerce")
    txt = s.astype(str).str.strip().str.lower().isin(["true", "yes", "y", "t", "event"])
    return ((num.fillna(0) > 0) | txt).astype(int)


def _percent(s):
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any() and num.max() <= 1.5:
        num = num * 100
    return num.clip(0, 100)


def _clean(name, df):
    if name == "campus_baseline" and "key" not in [_snake(c) for c in df.columns] and len(df) == 1:
        df = df.T.reset_index()
        df.columns = ["key", "value"]
    df = _canonical(df, ALIASES.get(name, {}))
    for tcol in ("timestamp", "date"):
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
            df = df.dropna(subset=[tcol]).sort_values(tcol).reset_index(drop=True)
    for c in ("consumption_kwh", "temperature_c", "rainfall_mm", "meals_served", "total_students", "occupancy",
              "capacity", "visitors", "bus", "two_wheeler", "car", "walk_cycle", "period", "intensity"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "attendance_rate" in df.columns:
        df["attendance_rate"] = _percent(df["attendance_rate"])
    if "event_flag" in df.columns:
        df["event_flag"] = _flag(df["event_flag"])
    if name == "electricity_data" and "occupancy" in df.columns:
        df["occupancy"] = pd.to_numeric(df["occupancy"], errors="coerce")
    return df


def load_all(upload_items: tuple = (), data_dir: Path = DATA_DIR):
    """upload_items: tuple of (dataset_name, csv_bytes). Uploads override files in data/.
    Returns (data: dict[name -> DataFrame | None], report: list[dict])."""
    uploads = dict(upload_items)
    data, report = {}, []
    for name in DATASETS:
        df, source, issue = None, "missing", ""
        try:
            if name in uploads:
                df, source = pd.read_csv(io.BytesIO(uploads[name])), "upload"
            elif (data_dir / f"{name}.csv").exists():
                df, source = pd.read_csv(data_dir / f"{name}.csv"), "data/"
        except Exception as exc:  # unreadable CSV
            df, issue = None, f"cannot read CSV: {exc}"
        if df is not None:
            df = _clean(name, df)
            missing = [c for c in REQUIRED[name] if c not in df.columns]
            if missing:
                issue, df = f"missing required column(s): {missing} - dataset ignored", None
            elif df.empty:
                issue, df = "no valid rows after cleaning - dataset ignored", None
        data[name] = df
        report.append(dict(dataset=name, source=source if df is not None else ("invalid" if issue else "missing"),
                           rows=0 if df is None else len(df), note=issue or ("ok" if df is not None else "using assumptions")))
    attach_context(data)
    return data, report


def attach_context(data):
    """Add temperature/event_flag columns to electricity & canteen data from weather/events files."""
    w, ev = data.get("weather_data"), data.get("campus_events")
    ev_dates = set(ev["date"].dt.normalize()) if ev is not None else set()
    for name, tcol in (("electricity_data", "timestamp"), ("canteen_data", "date")):
        df = data.get(name)
        if df is None:
            continue
        df = df.copy()
        df["date"] = df[tcol].dt.normalize()
        if ("temperature_c" not in df.columns or df["temperature_c"].isna().all()) and w is not None:
            daily_t = w.groupby(w["date"].dt.normalize())["temperature_c"].mean()
            df["temperature_c"] = df["date"].map(daily_t)
        if "event_flag" not in df.columns:
            df["event_flag"] = df["date"].isin(ev_dates).astype(int)
        data[name] = df


# ------------------------------ assumptions --------------------------------
def _tcol(df):
    return "timestamp" if "timestamp" in df.columns else "date"


def build_assumptions(data):
    """DEFAULT assumptions <- campus_baseline.csv <- light calibration from datasets
    (calibration only fills keys NOT given in campus_baseline.csv). Returns (A, notes)."""
    A, given, notes = dict(DEFAULT_VALUES), set(), []
    base = data.get("campus_baseline")
    if base is not None:
        for _, r in base.iterrows():
            k, v = str(r["key"]).strip(), pd.to_numeric(r["value"], errors="coerce")
            if k in A and pd.notna(v):
                A[k] = float(v)
                given.add(k)
        notes.append(f"{len(given)} assumption(s) loaded from campus_baseline.csv")

    def setk(k, v, why):
        if k not in given and np.isfinite(v):
            A[k] = float(v)
            notes.append(f"{k} = {v:,.3g} ({why})")

    elec, canteen, weather = data.get("electricity_data"), data.get("canteen_data"), data.get("weather_data")
    if canteen is not None and "total_students" in canteen:
        setk("total_students", float(canteen["total_students"].dropna().median()), "median of canteen_data")
    # typical attendance (weekday median) and temperature
    for df in (canteen, elec):
        if df is not None and "attendance_rate" in df and df["attendance_rate"].notna().any():
            wd = df[df[_tcol(df)].dt.dayofweek < 5]["attendance_rate"].dropna()
            if len(wd):
                setk("typical_attendance_pct", float(wd.median()), "weekday median in data")
                break
    temps = [d["temperature_c"].dropna() for d in (weather, elec, canteen) if d is not None and "temperature_c" in d]
    if temps and temps[0].size:
        setk("typical_temperature_c", float(temps[0].median()), "median temperature in data")

    typ_att = A["typical_attendance_pct"] / 100
    p_typ = A["total_students"] * typ_att
    # electricity calibration so the "Normal" scenario reproduces a typical weekday
    if elec is not None:
        t = elec[_tcol(elec)]
        daily = elec.groupby(t.dt.normalize())["consumption_kwh"].sum()
        daily = daily[daily > 0]
        if len(daily) >= 7:
            hourly = t.dt.hour.nunique() > 1
            wd_med = float(daily[daily.index.dayofweek < 5].median())
            base = float(daily.quantile(0.10) * 0.9)
            dyn = max(0.0, wd_med - base)
            setk("electricity_capacity_kwh", round(float(daily.max()) * 1.15, -2), "1.15 x max daily in data")
            peak_kw = float(elec["consumption_kwh"].max()) * 1.15 if hourly else \
                float(daily.max()) / A["operating_hours"] * 1.45 * 1.15
            # make sure the formula-based normal-day peak sits at <= ~75% of peak capacity
            cu = A["peak_periods"] / A["periods_per_day"]
            exc = max(0.0, A["typical_peak_concentration_pct"] / 100 - cu) / max(1 - cu, 1e-9)
            normal_peak = wd_med / A["operating_hours"] * (1 + A["peak_base_uplift"] + A["peak_conc_uplift"] * exc)
            peak_kw = max(peak_kw, normal_peak / 0.75)
            setk("peak_capacity_kw", round(peak_kw, -1), "1.15 x max hourly load, at least normal peak / 0.75")
            setk("base_load_kwh", base, "low-demand-day level in data")
            thr, typ_t = A["cooling_threshold_c"], A["typical_temperature_c"]
            cool_share = 0.4 if typ_t > thr else 0.0
            if cool_share:
                setk("cooling_kwh_per_degree", cool_share * dyn / ((typ_t - thr) * (0.4 + 0.6 * typ_att)),
                     "calibrated to typical weekday")
            n_typ = A["typical_simultaneous_classes"] * A["periods_per_day"]
            setk("kwh_per_present_student", (1 - cool_share) * 0.7 * dyn / max(p_typ, 1), "calibrated to typical weekday")
            setk("kwh_per_class_session", (1 - cool_share) * 0.3 * dyn / max(n_typ, 1), "calibrated to typical weekday")
    if canteen is not None:
        meals = canteen["meals_served"].dropna()
        meals = meals[meals > 0]
        if len(meals) >= 7:
            wd_meals = canteen[canteen[_tcol(canteen)].dt.dayofweek < 5]["meals_served"].dropna()
            typ = float(wd_meals.median()) if len(wd_meals) else float(meals.median())
            setk("canteen_capacity_meals", round(float(meals.max()) * 1.15, -1), "1.15 x max daily meals in data")
            setk("canteen_participation", float(np.clip(typ / max(p_typ * A["meals_per_participant"], 1), 0.05, 1.0)),
                 "calibrated to typical weekday meals")
    # commuting mode shares
    com = data.get("commuting_data")
    if com is not None:
        modes = ["bus", "two_wheeler", "car", "walk_cycle"]
        sub = com
        if "rainfall_mm" in com and (com["rainfall_mm"].fillna(0) < 1).sum() >= 3:
            sub = com[com["rainfall_mm"].fillna(0) < 1]
        vals = sub[modes].clip(lower=0).fillna(0)
        rs = vals.sum(axis=1).replace(0, np.nan)
        shares = vals.div(rs, axis=0).mean()
        if shares.notna().all():
            for m in modes:
                setk(f"share_{m}", float(shares[m]), "mean mode share on dry days in commuting_data")
    return A, notes


def reference_inputs(A) -> ScenarioInputs:
    """The 'Normal campus' scenario built from the baseline assumptions."""
    return ScenarioInputs(
        enrolled_students=A["total_students"], intake_growth_pct=0.0,
        attendance_rate=A["typical_attendance_pct"], temperature_c=A["typical_temperature_c"],
        rainfall_mm=0.0, simultaneous_classes=A["typical_simultaneous_classes"],
        peak_concentration_pct=A["typical_peak_concentration_pct"], event_day=False,
        event_visitors=0, event_intensity=2, day_of_week=2,
        allocation_efficiency_pct=A["allocation_efficiency_pct"],
    )


def finalize_assumptions(A):
    """If canteen_planned_meals is 0 (auto) set it to normal-day demand + buffer."""
    from simulation import simulate
    A = dict(A)
    if _planned_is_auto(A):
        normal = simulate(reference_inputs(A), {**A, "canteen_planned_meals": 0})
        A["canteen_planned_meals"] = normal["canteen"]["demand"] * (1 + A["canteen_prep_buffer"])
    return A


def _planned_is_auto(A):
    return not A.get("canteen_planned_meals")
