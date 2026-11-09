"""FoodFlow engine: read price datasets, check them, train, evaluate honestly, forecast.

Design rules (why the code looks like this):
- The target is the % CHANGE in price (log change), never the raw price.
  Raw prices let a model "win" by copying last month. A change means the
  same thing in 2010 and 2026.
- All features use shift(1) or older -> the model only sees the past
  (no data leakage).
- Evaluation is walk-forward: train on the past, test on the next months,
  repeat. Always compared against the baseline "same as last month".
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------------- column mapping
CANON = {
    "date":      ["date", "month", "period", "mp_month", "time", "day"],
    "market":    ["market", "mkt_name", "admin2", "district", "city", "town", "location", "region", "admin1"],
    "commodity": ["commodity", "cm_name", "food", "item", "product", "crop"],
    "price":     ["price", "mp_price", "value", "avg_price", "amount"],
    "unit":      ["unit", "um_name", "uom"],
    "pricetype": ["pricetype", "price_type", "pt_name", "type"],
    "latitude":  ["latitude", "lat", "mkt_latitude"],
    "longitude": ["longitude", "lon", "lng", "long", "mkt_longitude"],
    "currency":  ["currency", "cur_name"],
}
REQUIRED = ["date", "market", "commodity", "price"]


def map_columns(df: pd.DataFrame) -> dict:
    """Guess which column plays which role. Exact name first, then contains."""
    cols = {c.lower().strip(): c for c in df.columns}
    out = {}
    for role, names in CANON.items():
        hit = None
        for n in names:
            if n in cols:
                hit = cols[n]
                break
        if hit is None:
            for n in names:
                for lc, orig in cols.items():
                    if n in lc and orig not in out.values():
                        hit = orig
                        break
                if hit:
                    break
        if hit is not None:
            out[role] = hit
    return out


def load_price_file(path: str) -> tuple[pd.DataFrame, dict, list[str]]:
    """Read one CSV/Excel of prices. Returns (clean df, mapping, notes)."""
    notes = []
    if str(path).lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, low_memory=False)
    mapping = map_columns(raw)
    missing = [r for r in REQUIRED if r not in mapping]
    if missing:
        return raw, mapping, [f"MISSING:{m}" for m in missing]
    df = pd.DataFrame()
    for role, col in mapping.items():
        df[role] = raw[col]
    # WFP files have a 2nd row of "#date" tags -> becomes unparseable, dropped here
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    if bad_dates:
        notes.append(f"dropped {bad_dates} rows with unreadable dates")
    df = df.dropna(subset=["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    bad_price = int(df["price"].isna().sum() + (df["price"] <= 0).sum())
    if bad_price:
        notes.append(f"dropped {bad_price} rows with missing/zero prices")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    for c in ("market", "commodity"):
        df[c] = df[c].astype(str).str.strip()
    return df, mapping, notes


def _modal_filter(df: pd.DataFrame, col: str, notes: list[str], label: str) -> pd.DataFrame:
    """Keep each commodity's most common value of `col` (its modal unit, price type...)."""
    if col not in df.columns or df[col].isna().all():
        return df
    keep = df.groupby("commodity")[col].transform(lambda s: s.eq(s.mode().iat[0]) if len(s.mode()) else True)
    dropped = int((~keep).sum())
    if dropped:
        notes.append(f"kept each food's usual {label} only (dropped {dropped} rows in other {label}s)")
    return df[keep]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    df = _modal_filter(df, "pricetype", notes, "price type")
    df = _modal_filter(df, "unit", notes, "unit")
    df = _modal_filter(df, "currency", notes, "currency")
    # impossible prices: >8x away from that series' median is a typing mistake
    med = df.groupby(["commodity", "market"])["price"].transform("median")
    bad = (df["price"] < med / 8) | (df["price"] > med * 8)
    if bad.sum():
        notes.append(f"dropped {int(bad.sum())} impossible prices (8x away from that market's normal)")
    return df[~bad].copy(), notes


# ---------------------------------------------------------------- panel + features
def build_panel(df: pd.DataFrame, foods: list[str] | None = None, min_months: int = 24,
                extras: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per food x market x month, every month present (gaps = NaN)."""
    if foods:
        df = df[df["commodity"].isin(foods)]
    agg = {"price": "mean"}
    has_geo = "latitude" in df.columns and "longitude" in df.columns
    g = df.groupby(["commodity", "market", "month"], as_index=False).agg(agg)
    if has_geo:
        geo = df.groupby("market")[["latitude", "longitude"]].first().reset_index()
        g = g.merge(geo, on="market", how="left")
    counts = g.groupby(["commodity", "market"])["month"].transform("count")
    g = g[counts >= min_months]
    out = []
    for (food, mk), s in g.groupby(["commodity", "market"]):
        idx = pd.date_range(s["month"].min(), s["month"].max(), freq="MS")
        s = s.set_index("month").reindex(idx)
        s["commodity"], s["market"] = food, mk
        if has_geo:
            s[["latitude", "longitude"]] = s[["latitude", "longitude"]].ffill().bfill()
        s.index.name = "month"
        out.append(s.reset_index())
    panel = pd.concat(out, ignore_index=True).sort_values(["commodity", "market", "month"])
    # national (whole-dataset) median per food per month
    nat = g.groupby(["commodity", "month"])["price"].median().rename("nat").reset_index()
    panel = panel.merge(nat, on=["commodity", "month"], how="left")
    # optional extra tables (weather, fuel...) joined on market+month or month
    for ex in (extras or []):
        keys = ["market", "month"] if "market" in ex.columns else ["month"]
        panel = panel.merge(ex, on=keys, how="left")
    return panel


def make_features(panel: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, list[str]]:
    p = panel.copy()
    g = p.groupby(["commodity", "market"])
    p["last_price"] = g["price"].shift(1)
    p["c1"] = np.log(p["last_price"] / g["price"].shift(2))
    p["c3"] = np.log(p["last_price"] / g["price"].shift(4))
    p["c12"] = np.log(p["last_price"] / g["price"].shift(13))
    p["vs_nat"] = np.log(p["last_price"] / g["nat"].shift(1))
    p["nat_c"] = np.log(g["nat"].shift(1) / g["nat"].shift(2))
    tgt_month = (p["month"] + pd.DateOffset(months=horizon - 1)).dt.month
    p["ms"], p["mc"] = np.sin(2 * np.pi * tgt_month / 12), np.cos(2 * np.pi * tgt_month / 12)
    p["food_id"] = p["commodity"].astype("category").cat.codes
    p["refugee"] = p["market"].str.contains("refugee", case=False).astype(int)
    feats = ["c1", "c3", "c12", "vs_nat", "nat_c", "ms", "mc", "food_id", "refugee"]
    for c in ("latitude", "longitude"):
        if c in p.columns:
            feats.append(c)
    # extra numeric columns (weather, fuel...) -> use LAST month's value (no leakage)
    base = {"price", "nat", "last_price", "latitude", "longitude"}
    for c in panel.columns:
        if c in base or c in ("month", "commodity", "market"):
            continue
        if pd.api.types.is_numeric_dtype(panel[c]):
            p[f"x_{c}"] = g[c].shift(1)
            feats.append(f"x_{c}")
    # target: log change from last known price to the price `horizon` months on
    p["target_price"] = g["price"].shift(-(horizon - 1))
    p["y"] = np.log(p["target_price"] / p["last_price"])
    return p, feats


# ---------------------------------------------------------------- models + honest exam
def _gb(**kw):
    return HistGradientBoostingRegressor(
        loss=kw.pop("loss", "absolute_error"), max_depth=2, learning_rate=0.03,
        max_iter=300, min_samples_leaf=40, l2_regularization=1.0, random_state=0, **kw)


def walk_forward(data: pd.DataFrame, feats: list[str], horizon: int,
                 n_folds: int = 4, fold_months: int = 6) -> pd.DataFrame:
    """Train on past, test on next `fold_months`, slide forward. The fair exam."""
    d = data.dropna(subset=["y"])
    last = d["month"].max()
    rows = []
    for k in range(n_folds, 0, -1):
        start = last - pd.DateOffset(months=k * fold_months - 1)
        end = start + pd.DateOffset(months=fold_months)
        te = d[(d["month"] >= start) & (d["month"] < end)]
        tr = d[d["month"] + pd.DateOffset(months=horizon - 1) < start]
        if len(te) < 20 or len(tr) < 200:
            continue
        m = _gb().fit(tr[feats], tr["y"])
        pred = m.predict(te[feats])
        err_m = np.abs(te["last_price"] * np.exp(pred) - te["target_price"])
        err_b = np.abs(te["last_price"] - te["target_price"])
        td, ps = np.sign(te["y"].values), np.sign(pred)
        both = (td != 0) & (ps != 0)  # months where the model dared pick up/down
        rows.append({
            "test_window": f"{start:%Y-%m} to {end - pd.DateOffset(months=1):%Y-%m}",
            "rows": len(te),
            "baseline_MAE": round(err_b.mean(), 1),
            "model_MAE": round(err_m.mean(), 1),
            "model_wins": bool(err_m.mean() < err_b.mean()),
            "took_a_side": f"{both.mean():.0%}",
            "right_when_it_did": f"{(ps == td)[both].mean():.0%}" if both.any() else "-",
        })
    return pd.DataFrame(rows)


def fit_final(data: pd.DataFrame, feats: list[str]) -> dict:
    d = data.dropna(subset=["y"])
    # pick the shrink (how bold the moves are) on the last 12 target months
    cut = d["month"].max() - pd.DateOffset(months=11)
    tr, va = d[d["month"] < cut], d[d["month"] >= cut]
    m = _gb().fit(tr[feats], tr["y"])
    pv = m.predict(va[feats])
    shrinks = {s: np.abs(va["last_price"] * np.exp(s * pv) - va["target_price"]).mean()
               for s in (0.25, 0.5, 0.75, 1.0)}
    shrink = min(shrinks, key=shrinks.get)
    model = _gb().fit(d[feats], d["y"])
    lo = _gb(loss="quantile", quantile=0.10).fit(d[feats], d["y"])
    hi = _gb(loss="quantile", quantile=0.90).fit(d[feats], d["y"])
    return {"model": model, "lo": lo, "hi": hi, "shrink": float(shrink), "feats": feats}


def importances(bundle: dict, data: pd.DataFrame, n: int = 400) -> pd.Series:
    from sklearn.inspection import permutation_importance
    d = data.dropna(subset=["y"]).tail(n * 4)
    d = d.sample(min(n * 4, len(d)), random_state=0)
    r = permutation_importance(bundle["model"], d[bundle["feats"]], d["y"],
                               n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")
    return pd.Series(r.importances_mean, index=bundle["feats"]).sort_values(ascending=False)


def forecast_next(panel: pd.DataFrame, bundle: dict, horizon: int = 1) -> pd.DataFrame:
    """Predict the month after the newest price, per food x market."""
    latest = panel.dropna(subset=["price"])["month"].max()
    future = []
    for (food, mk), s in panel.groupby(["commodity", "market"]):
        if latest not in set(s["month"]) or s.loc[s["month"] == latest, "price"].isna().all():
            continue
        row = {"commodity": food, "market": mk, "month": latest + pd.DateOffset(months=1)}
        for c in ("latitude", "longitude"):
            if c in s.columns:
                row[c] = s[c].iloc[-1]
        future.append(row)
    if not future:
        return pd.DataFrame()
    ext = pd.concat([panel, pd.DataFrame(future)], ignore_index=True).sort_values(
        ["commodity", "market", "month"])
    # recompute nat so future rows inherit lagged values
    featd, feats = make_features(ext, horizon=horizon)
    fut = featd[featd["month"] == latest + pd.DateOffset(months=1)].copy()
    feats = bundle["feats"]
    chg = bundle["shrink"] * bundle["model"].predict(fut[feats])
    fut["forecast"] = (fut["last_price"] * np.exp(chg)).round(0)
    fut["low"] = (fut["last_price"] * np.exp(bundle["lo"].predict(fut[feats]))).round(0)
    fut["high"] = (fut["last_price"] * np.exp(bundle["hi"].predict(fut[feats]))).round(0)
    fut["trend"] = np.where(fut["forecast"] > fut["last_price"] * 1.01, "up",
                   np.where(fut["forecast"] < fut["last_price"] * 0.99, "down", "steady"))
    cols = ["commodity", "market", "month", "last_price", "forecast", "low", "high", "trend"]
    return fut[cols].sort_values(["commodity", "market"]).reset_index(drop=True)


def anomalies(data: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Months whose move was > k x that series' usual wobble (unsupervised alarm)."""
    d = data.copy()
    wob = d.groupby(["commodity", "market"])["y"].transform(
        lambda s: s.rolling(24, min_periods=12).std().shift(1))
    hit = d[(d["y"].abs() > k * wob) & d["y"].notna()]
    out = hit[["month", "commodity", "market", "last_price", "target_price"]].copy()
    out["change_pct"] = ((np.exp(hit["y"]) - 1) * 100).round(0)
    return out.sort_values("month", ascending=False)


def market_groups(data: pd.DataFrame, n_groups: int = 4) -> pd.DataFrame:
    """K-means on behaviour (unsupervised): price level, jumpiness, season strength."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    prof = data.dropna(subset=["y"]).groupby("market").agg(
        level=("vs_nat", "mean"), jumpiness=("y", "std"),
        months=("y", "count")).dropna()
    if len(prof) < n_groups + 1:
        return pd.DataFrame()
    Z = StandardScaler().fit_transform(prof[["level", "jumpiness"]])
    prof["group"] = KMeans(n_clusters=n_groups, n_init=10, random_state=0).fit_predict(Z)
    return prof.reset_index()
