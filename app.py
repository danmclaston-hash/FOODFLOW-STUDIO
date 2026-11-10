"""FoodFlow Studio - local web app around the engine.

Run:  uvicorn app:app --port 8000     then open http://localhost:8000
This is a single-user local tool: state lives in memory + ./workspace.
"""
from __future__ import annotations
import io, os, json, traceback
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from engine import pipeline as P
from engine import doctor as D
from engine import analyst as A
from engine import publish as PUB

app = FastAPI(title="FoodFlow Studio")
os.makedirs("workspace", exist_ok=True)

STATE: dict = {"prices": None, "extras": [], "report": None, "panel": None,
               "data": None, "feats": None, "eval": None, "bundle": None,
               "forecast": None, "anomalies": None, "importances": None, "groups": None}


def _err(e: Exception):
    traceback.print_exc()
    return JSONResponse({"error": str(e)[:300]}, status_code=400)


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    try:
        reports, extras = [], []
        for f in files:
            path = os.path.join("workspace", f.filename)
            with open(path, "wb") as out:
                out.write(await f.read())
            df, mapping, notes = P.load_price_file(path)
            if "price" in mapping and "commodity" in mapping:
                df, cn = P.clean(df)
                STATE["prices"] = df
                rep = D.examine(df, mapping, notes + cn)
                rep["file"] = f.filename
                rep["role"] = "prices"
                STATE["report"] = rep
                reports.append(rep)
            else:
                # extra table (weather, fuel, exchange rate...): join on month (+market)
                raw = pd.read_csv(path, low_memory=False)
                cols = {c.lower(): c for c in raw.columns}
                date_col = next((cols[k] for k in ("month", "date", "period") if k in cols), None)
                if date_col is None:
                    reports.append({"file": f.filename, "role": "ignored", "ok": False,
                                    "problems": ["No date/month column - I can't join this table."]})
                    continue
                raw["month"] = pd.to_datetime(raw[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
                raw = raw.dropna(subset=["month"])
                keep = ["month"] + ([cols["market"]] if "market" in cols else [])
                num = [c for c in raw.columns if c not in keep + [date_col]
                       and pd.api.types.is_numeric_dtype(raw[c])]
                ex = raw[keep + num].rename(columns={cols.get("market", ""): "market"})
                ex = ex.groupby([c for c in ("market", "month") if c in ex.columns], as_index=False).mean(numeric_only=True)
                STATE["extras"].append(ex)
                reports.append({"file": f.filename, "role": "extra clues", "ok": True,
                                "fixes": [f"Joined on {'market+month' if 'market' in ex.columns else 'month'}; "
                                          f"numeric clues: {', '.join(num[:8])}"], "problems": [], "questions": []})
        return {"reports": reports}
    except Exception as e:
        return _err(e)


@app.post("/api/train")
async def train(cfg: dict):
    try:
        if STATE["prices"] is None:
            return JSONResponse({"error": "Upload a price dataset first."}, status_code=400)
        foods = cfg.get("foods") or None
        horizon = int(cfg.get("horizon", 1))
        panel = P.build_panel(STATE["prices"], foods=foods, min_months=int(cfg.get("min_months", 24)),
                              extras=STATE["extras"])
        data, feats = P.make_features(panel, horizon=horizon)
        ev = P.walk_forward(data, feats, horizon=horizon)
        bundle = P.fit_final(data, feats)
        imp = P.importances(bundle, data)
        fc = P.forecast_next(panel, bundle, horizon=horizon)
        an = P.anomalies(data)
        gr = P.market_groups(data)
        STATE.update(panel=panel, data=data, feats=feats, eval=ev, bundle=bundle,
                     forecast=fc, anomalies=an, importances=imp, groups=gr)
        fc.to_csv("workspace/forecast.csv", index=False)
        pub = {"published": False}
        if PUB.enabled():
            try:
                pub = PUB.publish(STATE, dataset=cfg.get("dataset", "upload"), horizon=horizon)
            except Exception as e:
                pub = {"published": False, "reason": str(e)[:200]}
        return {
            "published": pub,
            "eval": ev.to_dict("records"),
            "importances": {k: round(float(v), 5) for k, v in imp.head(10).items()},
            "n_rows": int(data["y"].notna().sum()),
            "n_series": int(data.groupby(["commodity", "market"]).ngroups),
            "foods": sorted(panel["commodity"].unique().tolist()),
            "shrink": bundle["shrink"],
            "forecast_month": fc["month"].iloc[0].strftime("%B %Y") if len(fc) else None,
            "anomalies": an.head(12).assign(month=an["month"].dt.strftime("%Y-%m")).head(12).to_dict("records"),
            "groups": gr.round(3).to_dict("records") if len(gr) else [],
        }
    except Exception as e:
        return _err(e)


@app.get("/api/forecast")
async def forecast():
    fc = STATE.get("forecast")
    if fc is None:
        return JSONResponse({"error": "Train first."}, status_code=400)
    out = fc.copy()
    out["month"] = out["month"].dt.strftime("%Y-%m")
    return {"rows": out.to_dict("records")}


@app.get("/api/forecast.csv")
async def forecast_csv():
    return FileResponse("workspace/forecast.csv", filename="forecast.csv")


@app.post("/api/ask")
async def ask(q: dict):
    try:
        return A.ask(str(q.get("question", "")), STATE)
    except Exception as e:
        return _err(e)


@app.post("/api/publish")
async def publish_now():
    try:
        if STATE.get("forecast") is None:
            return JSONResponse({"error": "Train first."}, status_code=400)
        return PUB.publish(STATE, dataset="manual", horizon=1)
    except Exception as e:
        return _err(e)


@app.get("/api/foods")
async def foods():
    if STATE["prices"] is None:
        return {"foods": []}
    vc = STATE["prices"]["commodity"].value_counts()
    return {"foods": [{"name": k, "rows": int(v)} for k, v in vc.items()]}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
