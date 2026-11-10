"""Publish trained results to Supabase so the public dashboard can show them.

The brain (this app) runs on your PC and WRITES with the service key.
The dashboard (Cloudflare Pages) only READS with the anon key.

Set these before starting the app (never commit the service key):
    SUPABASE_URL=https://fdrzjxyhxwwkordscicn.supabase.co
    SUPABASE_SERVICE_KEY=<Project Settings -> API -> service_role key>
"""
from __future__ import annotations
import os
import pandas as pd
import requests


def _cfg():
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    return url.rstrip("/"), {"apikey": key, "Authorization": f"Bearer {key}",
                             "Content-Type": "application/json", "Prefer": "return=representation"}


def enabled() -> bool:
    return _cfg() is not None


def publish(state: dict, dataset: str, horizon: int) -> dict:
    cfg = _cfg()
    if cfg is None:
        return {"published": False, "reason": "SUPABASE_URL / SUPABASE_SERVICE_KEY not set"}
    url, h = cfg
    ev, imp, fc, an, data = (state["eval"], state["importances"], state["forecast"],
                             state["anomalies"], state["data"])
    run = {"dataset": dataset, "horizon": horizon,
           "n_rows": int(data["y"].notna().sum()),
           "n_series": int(data.groupby(["commodity", "market"]).ngroups),
           "exam": ev.to_dict("records"),
           "importances": {k: round(float(v), 5) for k, v in imp.head(10).items()}}
    r = requests.post(f"{url}/rest/v1/runs", json=run, headers=h, timeout=60)
    r.raise_for_status()
    run_id = r.json()[0]["id"]

    def rows(df: pd.DataFrame, cols: list[str]) -> list[dict]:
        out = df[cols].copy()
        out["month"] = out["month"].dt.strftime("%Y-%m-%d")
        out["run_id"] = run_id
        return out.where(pd.notna(out), None).to_dict("records")

    for i in range(0, len(fc), 500):
        requests.post(f"{url}/rest/v1/forecasts", headers=h, timeout=60,
                      json=rows(fc.iloc[i:i + 500], ["commodity", "market", "month", "last_price",
                                                      "forecast", "low", "high", "trend"])).raise_for_status()
    if len(an):
        requests.post(f"{url}/rest/v1/shocks", headers=h, timeout=60,
                      json=rows(an.head(100), ["month", "commodity", "market", "change_pct"])).raise_for_status()
    return {"published": True, "run_id": run_id, "forecasts": int(len(fc)), "shocks": int(min(len(an), 100))}
