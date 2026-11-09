"""The Data Doctor: looks at uploaded data, reports problems in plain words,
and asks the user the questions a careful analyst would ask."""
from __future__ import annotations
import pandas as pd
from .pipeline import REQUIRED


def examine(df: pd.DataFrame, mapping: dict, notes: list[str]) -> dict:
    report = {"ok": True, "found_columns": mapping, "fixes": list(notes),
              "problems": [], "questions": [], "facts": {}}
    missing = [r for r in REQUIRED if r not in mapping]
    if missing:
        report["ok"] = False
        report["problems"].append(
            "I could not find these required columns: " + ", ".join(missing) +
            ". I need date, market/place, food name, and price. "
            "Rename your columns or upload a file that has them.")
        return report

    report["facts"] = {
        "rows": int(len(df)),
        "markets": int(df["market"].nunique()),
        "foods": int(df["commodity"].nunique()),
        "from": str(df["month"].min().date()),
        "to": str(df["month"].max().date()),
        "top_foods": df["commodity"].value_counts().head(12).to_dict(),
    }

    months = df["month"].nunique()
    if months < 36:
        report["problems"].append(
            f"Only {months} months of data. The model needs 3+ years to learn seasons. "
            "It will still run, but expect weak results.")
    series = df.groupby(["commodity", "market"])["month"].nunique()
    good = int((series >= 24).sum())
    if good == 0:
        report["ok"] = False
        report["problems"].append(
            "No food+market pair has 24+ months of history. There is nothing to learn from.")
    else:
        report["facts"]["usable_series"] = good

    if "latitude" in df.columns and df["latitude"].notna().any():
        lat_span = df["latitude"].max() - df["latitude"].min()
        lon_span = df["longitude"].max() - df["longitude"].min()
        if max(lat_span, lon_span) > 15:
            report["questions"].append(
                "Your markets are very far apart (over ~1500 km). Is this really one "
                "area? Mixing countries is fine ONLY if that is what you intend.")
    else:
        report["fixes"].append("No GPS columns found - the model will run without location clues.")

    if "unit" not in mapping:
        report["questions"].append(
            "No unit column found. Are ALL prices in the same unit (e.g. per kg)? "
            "Mixing cups and sacks would confuse the model.")
    if "currency" in df.columns and df["currency"].nunique() > 1:
        report["problems"].append(
            "More than one currency in the file. I kept the most common one per food.")

    gaps = 1 - len(df.drop_duplicates(["commodity", "market", "month"])) / max(
        months * len(series), 1)
    report["questions"].append("Which foods should I model? (Leave empty = all with enough history.)")
    report["questions"].append("Forecast 1 month ahead (default) or 3 months ahead?")
    return report
