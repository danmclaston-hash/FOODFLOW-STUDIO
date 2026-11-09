"""Satellite weather connector (NASA POWER, free, no account).

For every market in your price file, downloads monthly rain + temperature
and writes weather.csv. Upload that CSV into FoodFlow Studio as an extra
table and retrain: the engine joins it on market+month automatically.

Run (needs internet):
    python connectors/nasa_power.py path/to/your_price_file.csv
Then upload weather.csv in the app.

If it fails: check your internet, then just run it again - it resumes
from the cache folder, nothing is downloaded twice.
"""
import json, os, sys, time
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import pipeline as P

URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"


def main(price_file: str, out: str = "weather.csv"):
    df, mapping, _ = P.load_price_file(price_file)
    if "latitude" not in df.columns:
        sys.exit("Your price file has no GPS columns, so I can't look up weather per market.")
    pts = df.groupby("market")[["latitude", "longitude"]].first().reset_index()
    y0, y1 = int(df["month"].dt.year.min()), int(df["month"].dt.year.max())
    os.makedirs("weather_cache", exist_ok=True)
    rows = []
    for i, p in pts.iterrows():
        path = f"weather_cache/{p.latitude:.2f}_{p.longitude:.2f}_{y0}_{y1}.json"
        if not os.path.exists(path):
            print(f"[{i+1}/{len(pts)}] {p.market} ...")
            r = requests.get(URL, params={
                "parameters": "PRECTOTCORR,T2M", "community": "AG",
                "latitude": p.latitude, "longitude": p.longitude,
                "start": y0, "end": y1, "format": "JSON"}, timeout=90)
            r.raise_for_status()
            json.dump(r.json(), open(path, "w"))
            time.sleep(1)  # be polite to NASA
        par = json.load(open(path))["properties"]["parameter"]
        for ym, rain in par["PRECTOTCORR"].items():
            if ym.endswith("13"):  # code for the yearly average, skip
                continue
            t = par["T2M"].get(ym, -999)
            rows.append((p.market, f"{ym[:4]}-{ym[4:6]}-01",
                         None if rain == -999 else rain, None if t == -999 else t))
    pd.DataFrame(rows, columns=["market", "month", "rain_mm_day", "temp_c"]).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows). Upload it in FoodFlow Studio and retrain.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python connectors/nasa_power.py your_price_file.csv")
    main(sys.argv[1])
