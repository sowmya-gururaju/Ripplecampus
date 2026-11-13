"""
OPTIONAL fallback: writes SYNTHETIC demo CSVs into data/ so the app runs before you have real data.
Never present these numbers as measured campus statistics. Run: python generate_sample_data.py
(campus_baseline.csv and dependencies.csv are NOT touched.)
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"
rng = np.random.default_rng(42)  # fixed seed -> reproducible
DAYS = 90
STUDENTS, ROOMS, CAP = 5000, 80, 60

dates = pd.date_range("2026-01-05", periods=DAYS, freq="D")
dow = dates.dayofweek.to_numpy()
weekday = dow < 5
temp = 27 + 5 * np.sin(np.arange(DAYS) / DAYS * np.pi) + rng.normal(0, 1.5, DAYS)
rain = np.where(rng.random(DAYS) < 0.2, rng.gamma(2, 6, DAYS), 0.0).round(1)
event = np.zeros(DAYS, int)
event[rng.choice(np.arange(5, DAYS), 9, replace=False)] = 1
att = np.where(weekday, rng.normal(76, 5, DAYS), rng.normal(12, 3, DAYS)) - rain * 0.2
att = np.clip(att, 3, 98).round(1)
visitors = event * rng.integers(800, 2500, DAYS)

pd.DataFrame({"date": dates.date, "temperature_c": temp.round(1), "rainfall_mm": rain, "is_synthetic": True}) \
    .to_csv(OUT / "weather_data.csv", index=False)
pd.DataFrame({"date": dates[event == 1].date, "event_name": "Campus event", "visitors": visitors[event == 1],
              "intensity": rng.integers(1, 4, event.sum()), "is_synthetic": True}) \
    .to_csv(OUT / "campus_events.csv", index=False)

# hourly electricity (kWh per hour)
rows = []
shape = np.exp(-((np.arange(24) - 13) ** 2) / (2 * 3.5 ** 2))  # daytime bell
for i, d in enumerate(dates):
    for h in range(24):
        occ = att[i] / 100 * shape[h]
        cool = max(0, temp[i] - 24) * 28 * (0.4 + 0.6 * att[i] / 100) * (0.3 + shape[h])
        evt = event[i] * 120 * shape[h]
        kwh = 520 + 1900 * occ + cool + evt + rng.normal(0, 40)
        rows.append((d + pd.Timedelta(hours=h), round(max(kwh, 200), 1), round(temp[i] + 3 * shape[h] - 1.5, 1),
                     att[i], event[i], True))
pd.DataFrame(rows, columns=["timestamp", "consumption_kwh", "temperature_c", "attendance_rate",
                            "event_flag", "is_synthetic"]).to_csv(OUT / "electricity_data.csv", index=False)

meals = STUDENTS * att / 100 * 0.62 * 1.5 + visitors * 1.2 + (temp - 28).clip(0) * 15 + rng.normal(0, 90, DAYS)
pd.DataFrame({"date": dates.date, "meals_served": meals.round(0), "attendance_rate": att, "event_flag": event,
              "total_students": STUDENTS, "temperature_c": temp.round(1), "is_synthetic": True}) \
    .to_csv(OUT / "canteen_data.csv", index=False)

# classroom: room x period for 20 weekdays
cl = []
period_load = np.array([.55, .85, 1.0, .95, .6, .75, .8, .35])  # peak mid-morning
for d in dates[weekday][:20]:
    for p in range(1, 9):
        for r in range(1, ROOMS + 1):
            if rng.random() < min(1, period_load[p - 1] * 0.8):
                occ = int(np.clip(rng.normal(CAP * period_load[p - 1] * 0.95, 12), 5, CAP * 1.3))
                cl.append((d.date(), p, f"R{r:03d}", occ, CAP))
pd.DataFrame(cl, columns=["date", "period", "room_id", "occupancy", "capacity"]).assign(is_synthetic=True) \
    .to_csv(OUT / "classroom_data.csv", index=False)

# commuting: daily mode counts of commuting students
com = []
for i in range(DAYS):
    n = STUDENTS * 0.7 * att[i] / 100
    r = min(1, rain[i] / 20)
    sh = np.array([0.30 + 0.25 * 0.3 * r, 0.30 * (1 - 0.4 * r), 0.15 + 0.1 * r, 0.25 * (1 - 0.25 * r)])
    sh = sh / sh.sum()
    com.append((dates[i].date(), *(sh * n).round(0).astype(int), rain[i], True))
pd.DataFrame(com, columns=["date", "bus", "two_wheeler", "car", "walk_cycle", "rainfall_mm", "is_synthetic"]) \
    .to_csv(OUT / "commuting_data.csv", index=False)
print("Synthetic demo data written to", OUT)
