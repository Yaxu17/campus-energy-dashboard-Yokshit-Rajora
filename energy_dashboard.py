import sys, logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------- Config ----------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------- Utilities & Ingestion ----------
def read_all_csvs(data_dir: Path) -> pd.DataFrame:
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        logging.warning("No CSV files found in data/. Generating sample data.")
        return generate_sample_data()
    frames = []
    for f in csvs:
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
            # heuristics for columns
            ts_col = None
            kwh_col = None
            for c in df.columns:
                lc = c.lower()
                if any(x in lc for x in ("time","date","timestamp")):
                    ts_col = c
                if any(x in lc for x in ("kwh","energy","usage","consump")):
                    kwh_col = c
            if ts_col is None:
                ts_col = df.columns[0]
            if kwh_col is None:
                # choose first numeric column that's not timestamp
                candidates = [c for c in df.columns if c!=ts_col and pd.api.types.is_numeric_dtype(df[c])]
                kwh_col = candidates[0] if candidates else (df.columns[1] if len(df.columns)>1 else None)
            if kwh_col is None:
                logging.error(f"No kWh-like column in {f.name}, skipping.")
                continue
            df = df.rename(columns={ts_col: "timestamp", kwh_col: "kwh"})
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df['kwh'] = pd.to_numeric(df['kwh'], errors='coerce').fillna(0)
            building = f.stem
            df['building'] = df.get('building', building)
            frames.append(df[['timestamp','kwh','building']])
            logging.info(f"Loaded {f.name} ({len(df)} rows).")
        except Exception as e:
            logging.error(f"Failed to read {f.name}: {e}")
    if not frames:
        return generate_sample_data()
    df_all = pd.concat(frames, ignore_index=True)
    return df_all

def generate_sample_data(days=14, buildings=('Library','Hostel','LabBlock')):
    rows = []
    now = pd.Timestamp.now().normalize()
    for b in buildings:
        for d in range(days):
            day = now - pd.Timedelta(days=days-d-1)
            for h in range(24):
                ts = day + pd.Timedelta(hours=h)
                base = 50 if b=='Library' else 80 if b=='Hostel' else 40
                variation = np.random.normal(loc=0, scale=8)
                kwh = max(0.1, base/24 + variation/10 + (18 <= h <= 21) * 3)
                rows.append({'timestamp': ts, 'kwh': round(kwh,3), 'building': b})
    df = pd.DataFrame(rows)
    logging.info(f"Generated sample data ({len(df)} rows).")
    return df

# ---------- Aggregation functions ----------
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')
    if 'building' not in df.columns:
        df['building'] = 'Unknown'
    return df

def calculate_daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby('building').resample('D').kwh.sum().reset_index()
    return daily

def calculate_weekly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    weekly = df.groupby('building').resample('W').kwh.sum().reset_index()
    return weekly

def building_wise_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby('building').kwh.agg(['sum','mean','min','max','std']).rename(columns={
        'sum':'total_kwh','mean':'mean_kwh','min':'min_kwh','max':'max_kwh','std':'std_kwh'
    }).fillna(0)
    grp = grp.reset_index()
    return grp

# ---------- OOP Model ----------
class MeterReading:
    def __init__(self, timestamp, kwh):
        self.timestamp = pd.to_datetime(timestamp)
        self.kwh = float(kwh)

class Building:
    def __init__(self, name):
        self.name = name
        self.readings = []
    def add_reading(self, mr: MeterReading):
        self.readings.append(mr)
    def total_consumption(self):
        return sum(r.kwh for r in self.readings)
    def peak_hour(self):
        if not self.readings:
            return None
        df = pd.DataFrame([{'timestamp':r.timestamp,'kwh':r.kwh} for r in self.readings]).set_index('timestamp')
        s = df.resample('H').kwh.sum()
        return s.idxmax(), s.max()
    def to_dict(self):
        return {'building': self.name, 'total_kwh': self.total_consumption()}

class BuildingManager:
    def __init__(self):
        self.buildings = {}
    def ingest_df(self, df: pd.DataFrame):
        for b, g in df.reset_index().groupby('building'):
            bm = Building(b)
            for _, row in g.iterrows():
                bm.add_reading(MeterReading(row['timestamp'], row['kwh']))
            self.buildings[b] = bm
    def summary(self):
        return pd.DataFrame([bm.to_dict() for bm in self.buildings.values()])

# ---------- Visualization ----------
def make_dashboard(daily_df: pd.DataFrame, weekly_df: pd.DataFrame, df_clean: pd.DataFrame, out_path: Path):
    plt.close('all')
    fig, axes = plt.subplots(3,1, figsize=(10,12), constrained_layout=True)
    # Trend Line – daily consumption per building
    for b, g in daily_df.groupby('building'):
        axes[0].plot(g['timestamp'], g['kwh'], marker='o', label=b)
    axes[0].set_title("Daily Consumption per Building")
    axes[0].set_ylabel("kWh")
    axes[0].legend(fontsize='small')
    # Bar Chart – compare average weekly usage across buildings
    avg_week = weekly_df.groupby('building').kwh.mean().sort_values(ascending=False)
    axes[1].bar(avg_week.index, avg_week.values)
    axes[1].set_title("Average Weekly Usage by Building")
    axes[1].set_ylabel("kWh (avg/week)")
    axes[1].tick_params(axis='x', rotation=45)
    # Scatter Plot – hourly consumption
    hourly = df_clean.groupby('building').resample('h').kwh.sum().reset_index()
    for b, g in hourly.groupby('building'):
        axes[2].scatter(g['timestamp'], g['kwh'], s=8, label=b)
    axes[2].set_title("Hourly Consumption (scatter)")
    axes[2].set_ylabel("kWh")
    axes[2].legend(fontsize='small')
    fig.suptitle("Campus Energy Dashboard", fontsize=14)
    plt.savefig(out_path, dpi=150)
    logging.info(f"Dashboard saved to {out_path}")

# ---------- Persistence & Summary ----------
def save_outputs(df_clean: pd.DataFrame, building_summary: pd.DataFrame, out_dir: Path):
    df_clean.reset_index().to_csv(out_dir/"cleaned_energy_data.csv", index=False)
    building_summary.to_csv(out_dir/"building_summary.csv", index=False)
    logging.info(f"Exported cleaned and summary CSVs to {out_dir}")

def generate_summary_text(df_clean: pd.DataFrame, building_summary: pd.DataFrame, out_dir: Path):
    total = df_clean.kwh.sum()
    top = building_summary.sort_values('total_kwh', ascending=False).iloc[0] if not building_summary.empty else {'building':'N/A','total_kwh':0}
    hourly = df_clean.resample('h').kwh.sum()
    peak_time = hourly.idxmax() if not hourly.empty else None
    peak_val = hourly.max() if not hourly.empty else 0
    lines = [
        f"Campus Energy Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        f"Total campus consumption: {total:.2f} kWh",
        f"Highest-consuming building: {top['building']} ({top['total_kwh']:.2f} kWh)",
        f"Peak campus load: {peak_val:.2f} kWh at {peak_time}",
        "",
        "Weekly & Daily trends: see output/dashboard.png for visuals."
    ]
    txt = "\n".join(lines)
    (out_dir/"summary.txt").write_text(txt)
    logging.info(f"Summary written to {out_dir/'summary.txt'}")
    return txt

# ---------- Main ----------
def main():
    df = read_all_csvs(DATA_DIR)
    df = prepare_df(df)
    daily = calculate_daily_totals(df)
    weekly = calculate_weekly_aggregates(df)
    summary = building_wise_summary(df)
    manager = BuildingManager()
    manager.ingest_df(df)
    save_outputs(df, summary, OUT_DIR)
    make_dashboard(daily, weekly, df, OUT_DIR/"dashboard.png")
    txt = generate_summary_text(df, summary, OUT_DIR)
    print(txt)

if __name__ == "__main__":
    main()
