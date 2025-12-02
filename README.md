# campus-energy-dashboard-<yourname>

Compact single-file Campus Energy Dashboard (Python).

## What's included
- `energy_dashboard.py` — single compact script implementing:
  - CSV ingestion from `data/`
  - Data cleaning & validation
  - Aggregations (daily/weekly)
  - OOP model (Building, MeterReading, BuildingManager)
  - Visual dashboard saved to `output/dashboard.png`
  - Exports: `output/cleaned_energy_data.csv`, `output/building_summary.csv`, `output/summary.txt`

## How to run
1. Put CSV files in the `data/` folder. Filenames will be used as building names.
2. Run:
```bash
python3 energy_dashboard.py
```
3. Outputs will be in `output/`.

## Notes
- If `data/` is empty, the script generates sample data so you can test immediately.
- This project was created based on the Capstone assignment spec provided by the user. See assignment file for task details.
