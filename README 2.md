# 🪞 MIRROR - Smart Campus Ripple Analytics
*One campus. Many decisions. Endless ripple effects.*

Hackathon track: **A - Data Science, DS01 Smart Campus Resource Analytics.**
Cascading-impact simulator + forecasting + scenario comparison + rule-based recommendations for
electricity, classrooms, canteen, water, buses and parking.

## Run
```bash
cd mirror
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# put your CSVs in data/ (see data/README.md)   -- or, for demo data only:
python generate_sample_data.py
streamlit run app.py
```

## Connecting datasets
1. Name files exactly as in `data/README.md` and place them in `data/` (or upload in sidebar -> "Data & CSV upload").
2. Open "Configurable assumptions & data status" in the app: it shows which files loaded, row counts, problems,
   and the calibration applied.
3. Edit `data/campus_baseline.csv` with your real campus numbers (students, rooms, buses, parking, water supply).
   Anything not given there is calibrated from your data where possible, else uses placeholder defaults.
4. If your column names differ, add them to `ALIASES` in `data_loader.py` (one line per name).

**Calibration:** electricity capacity, base load and per-student/per-session/cooling coefficients are fitted so the
"Normal campus" reproduces a typical weekday in your data; canteen capacity and participation likewise; commuting
mode shares come from commuting_data on dry days.

## Files
| File | Purpose |
|---|---|
| app.py | Streamlit dashboard: sidebar controls, 10 tabs |
| simulation.py | Transparent formulas; `simulate(inputs, assumptions)`; all assumptions + descriptions |
| graph_engine.py | NetworkX dependency graph, Plotly figure, active-ripple logic, edge verification |
| forecasting.py | Random Forest forecasts (electricity, canteen), MAE/RMSE, seasonal-naive baseline, classroom stats |
| scenario_analysis.py | Scenarios A/B/C, one-at-a-time interventions, factor contributions |
| recommendations.py | Rules on simulation outputs; finds smallest fix and re-simulates |
| data_loader.py | CSV loading, alias mapping, cleaning, calibration |
| generate_sample_data.py | OPTIONAL synthetic demo data (seeded) |

## How it works
- **Inputs:** population + intake growth, attendance, temperature, rainfall, timetable (classes per period, % in peak
  periods), campus event (visitors, intensity). **Commuting is an output.**
- Present students = total x attendance. Hostel residents do not commute (no double counting).
- Electricity = base + occupancy + class sessions + cooling (above threshold) + event; peak kW from timetable concentration.
- Classrooms: sessions per period from timetable shape; seat and room utilisation per period; overcrowded/under-used rooms.
- Canteen = present x participation x meals (+ heat uplift) + event meals; excess and waste vs capacity/preparation.
- Water = students + hostel + event + meal water, with heat uplift.
- Commuting: mode shares shifted by rain (configurable), bus demand in the peak window, parking in car-equivalent spaces,
  road pressure, congestion index, CO2 estimate.
- Graph edges all map to real calculations; "Dependency verification" re-runs the model per factor to prove it.
- Risk colours: <70 healthy, 70-90 warning, 90-100 critical, >100 overloaded.

## Team split (5 people, 8 hours)
1. **Data** - source/clean datasets, fill campus_baseline.csv, verify data status panel.
2. **Simulation** - review/tune assumptions and formulas in simulation.py, sanity-check outputs.
3. **ML** - forecasting.py evaluation, tab explanation, check baseline comparison.
4. **Graph + scenarios + recommendations** - graph_engine.py, scenario_analysis.py, recommendations.py.
5. **UI + pitch** - app.py polish, README, demo script, slides.

## 2-minute demo
1. (15s) Problem: systems analysed in isolation; MIRROR connects them. Show Overview.
2. (30s) Raise temperature + attendance, switch on an event; Butterfly tab: amber ripples, systems turn orange/red.
3. (20s) Commuting tab: rain shifts riders to buses/cars; bus and parking pressure rise.
4. (20s) Forecasting tab: RF vs baseline, MAE/RMSE (say whether data is real or synthetic).
5. (20s) Scenarios: A vs B vs C, bottleneck, best single fix.
6. (15s) Recommendations: what/why/action/simulated improvement.

## Assumptions & limitations
- Defaults are placeholders, not measured facts; coefficients are simple linear approximations.
- Commuting shift under rain uses configurable assumptions, not estimated from data.
- Forecast models use limited history; with synthetic data the evaluation only demonstrates the pipeline.
- Baseline forecast assumes regular hourly/daily rows (lag 168 h / 7 days).
- One representative day; no multi-day dynamics. Water "pressure" = utilisation level.
