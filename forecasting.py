"""
forecasting.py - demand forecasting + model evaluation + classroom descriptive analysis.

* RandomForestRegressor trained on the first 80% of the time-ordered data, evaluated on the
  last 20% (no shuffling -> no look-ahead leakage).
* Compared against a seasonal-naive baseline (same hour last week / same weekday last week).
* A second model is refit on all data and used to forecast the live scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


MIN_ROWS = 30


def _train(df, time_col, target, features, lag):
    d = df.dropna(subset=[target]).sort_values(time_col).reset_index(drop=True)
    d[features] = d[features].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=features)
    if len(d) < MIN_ROWS:
        raise ValueError(f"only {len(d)} usable rows (need >= {MIN_ROWS})")
    cut = int(len(d) * 0.8)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    params = dict(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
    eval_model = RandomForestRegressor(**params).fit(tr[features], tr[target])
    pred = eval_model.predict(te[features])
    y = te[target].to_numpy(float)
    series = d[target].to_numpy(float)
    idx = np.arange(cut, len(d))
    base = np.where(idx - lag >= 0, series[np.maximum(idx - lag, 0)], float(tr[target].mean()))

    def mae(p):
        return float(np.mean(np.abs(y - p)))

    def rmse(p):
        return float(np.sqrt(np.mean((y - p) ** 2)))

    metrics = dict(model_mae=mae(pred), model_rmse=rmse(pred), base_mae=mae(base), base_rmse=rmse(base),
                   mean_target=float(np.mean(y)))
    metrics["mae_improvement_pct"] = 100 * (metrics["base_mae"] - metrics["model_mae"]) / max(metrics["base_mae"], 1e-9)
    final = RandomForestRegressor(**params).fit(d[features], d[target])
    return dict(
        model=final, features=features, metrics=metrics, time_col=time_col, target=target,
        test=pd.DataFrame({"time": te[time_col].to_numpy(), "actual": y, "random_forest": pred, "baseline": base}),
        importance=pd.Series(final.feature_importances_, index=features).sort_values(),
        n_train=len(tr), n_test=len(te), lag=lag,
        train_medians=d[features].median().to_dict(),
        occ_ratio=float((d["occupancy"] / d["attendance_rate"].replace(0, np.nan)).median())
        if {"occupancy", "attendance_rate"} <= set(d.columns) else None,
    )


# ------------------------------- electricity --------------------------------
def train_electricity(df):
    d = df.copy()
    d["dayofweek"] = d["timestamp"].dt.dayofweek
    hourly = d["timestamp"].dt.hour.nunique() > 1
    feats = (["hour"] if hourly else []) + ["dayofweek"]
    if hourly:
        d["hour"] = d["timestamp"].dt.hour
    for c in ("temperature_c", "attendance_rate", "event_flag", "occupancy"):
        if c in d.columns and d[c].notna().mean() > 0.9:
            d[c] = d[c].fillna(d[c].median())
            feats.append(c)
    out = _train(d, "timestamp", "consumption_kwh", feats, lag=168 if hourly else 7)
    out.update(kind="electricity", hourly=hourly, unit="kWh")
    return out


def forecast_electricity_day(b, inp):
    """Forecast one scenario day -> dict(hourly DataFrame|None, daily_kwh, peak_hour, peak_kw)."""
    def row(hour=None):
        r = {"dayofweek": inp.day_of_week, "temperature_c": inp.temperature_c,
             "attendance_rate": inp.attendance_rate, "event_flag": int(bool(inp.event_day))}
        if hour is not None:
            r["hour"] = hour
        if "occupancy" in b["features"]:
            r["occupancy"] = inp.attendance_rate * (b["occ_ratio"] or 1.0)
        return {k: v for k, v in r.items() if k in b["features"]}

    if b["hourly"]:
        X = pd.DataFrame([row(h) for h in range(24)])[b["features"]]
        y = b["model"].predict(X)
        hourly = pd.DataFrame({"hour": range(24), "forecast_kwh": y})
        return dict(hourly=hourly, daily_kwh=float(y.sum()), peak_hour=int(np.argmax(y)), peak_kw=float(y.max()))
    y = float(b["model"].predict(pd.DataFrame([row()])[b["features"]])[0])
    return dict(hourly=None, daily_kwh=y, peak_hour=None, peak_kw=None)


# --------------------------------- canteen ----------------------------------
def train_canteen(df):
    d = df.copy()
    d["dayofweek"] = d["date"].dt.dayofweek
    feats = ["dayofweek"]
    for c in ("attendance_rate", "event_flag", "total_students", "temperature_c"):
        if c in d.columns and d[c].notna().mean() > 0.9:
            d[c] = d[c].fillna(d[c].median())
            feats.append(c)
    out = _train(d, "date", "meals_served", feats, lag=7)
    out.update(kind="canteen", hourly=False, unit="meals")
    return out


def forecast_canteen_day(b, inp):
    total = inp.enrolled_students * (1 + inp.intake_growth_pct / 100)
    r = {"dayofweek": inp.day_of_week, "attendance_rate": inp.attendance_rate,
         "event_flag": int(bool(inp.event_day)), "total_students": total, "temperature_c": inp.temperature_c}
    X = pd.DataFrame([{k: v for k, v in r.items() if k in b["features"]}])[b["features"]]
    return float(b["model"].predict(X)[0])


# ------------------------- classroom descriptive stats -----------------------
def analyze_classroom_history(df, A):
    """Per-period utilisation from classroom_data (rows = room x period [x date])."""
    d = df.copy()
    cap_default = A.get("classroom_capacity", 60)
    d["capacity"] = pd.to_numeric(d["capacity"], errors="coerce").fillna(cap_default) if "capacity" in d else cap_default
    d["util"] = d["occupancy"] / d["capacity"].replace(0, np.nan) * 100
    thr = A.get("underutilized_threshold_pct", 40)
    g = d.groupby("period")["util"]
    t = pd.DataFrame({
        "avg_util_pct": g.mean(),
        "overcrowded_share_pct": g.apply(lambda s: (s > 100).mean() * 100),
        "underutilized_share_pct": g.apply(lambda s: (s < thr).mean() * 100),
    }).reset_index()
    mean_u = float(d["util"].mean())
    t["suggested_shift_pct"] = np.where(t["avg_util_pct"] > mean_u,
                                        (t["avg_util_pct"] - mean_u) / t["avg_util_pct"].replace(0, np.nan) * 100, 0).round(1)
    peak = t.loc[t["avg_util_pct"].idxmax()]
    low = t.loc[t["avg_util_pct"].idxmin()]
    hot = t[t["avg_util_pct"] > mean_u * 1.15]["period"].tolist()
    cold = t[t["avg_util_pct"] < mean_u * 0.85]["period"].tolist()
    msg = (f"Peak utilisation is period {peak['period']:.0f} ({peak['avg_util_pct']:.0f}%), the quietest is period "
           f"{low['period']:.0f} ({low['avg_util_pct']:.0f}%). ")
    if hot and cold:
        msg += f"Moving some sessions from periods {[int(x) for x in hot]} to periods {[int(x) for x in cold]} would flatten the load."
    return dict(table=t.round(1), mean_util=mean_u, message=msg)
