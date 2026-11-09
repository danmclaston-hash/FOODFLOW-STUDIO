"""The Analyst: answers questions about the trained system in plain words.

Two brains:
1. If OLLAMA_URL is set (an open-source LLM running on YOUR computer, e.g.
   `ollama run llama3.2`), questions go there WITH the real numbers attached,
   so the answers are grounded in your data, not invented.
2. Otherwise a built-in rule brain answers from the computed results only.
   It never makes numbers up: if it doesn't know, it says so.
"""
from __future__ import annotations
import os, json
import pandas as pd


def _context(state: dict) -> str:
    """Everything true we know, as text the LLM (or rules) may use."""
    parts = []
    if state.get("report"):
        f = state["report"].get("facts", {})
        parts.append(f"Data: {f.get('rows')} rows, {f.get('markets')} markets, "
                     f"{f.get('foods')} foods, {f.get('from')} to {f.get('to')}.")
    if state.get("eval") is not None and len(state["eval"]):
        parts.append("Walk-forward exam (model vs 'same as last month' baseline):\n"
                     + state["eval"].to_string(index=False))
    if state.get("importances") is not None:
        top = state["importances"].head(5)
        parts.append("Most useful clues: " + ", ".join(f"{k} ({v:.4f})" for k, v in top.items()))
    if state.get("forecast") is not None and len(state["forecast"]):
        fc = state["forecast"]
        parts.append(f"Forecast for {fc['month'].iloc[0]:%B %Y}: {len(fc)} series. "
                     f"Up: {(fc['trend']=='up').sum()}, down: {(fc['trend']=='down').sum()}, "
                     f"steady: {(fc['trend']=='steady').sum()}.")
    if state.get("anomalies") is not None and len(state["anomalies"]):
        a = state["anomalies"].head(5)
        parts.append("Latest price shocks: " + "; ".join(
            f"{r.commodity} in {r.market} {r.month:%Y-%m} ({r.change_pct:+.0f}%)"
            for r in a.itertuples()))
    return "\n\n".join(parts) if parts else "Nothing trained yet."


def ask(question: str, state: dict) -> dict:
    ctx = _context(state)
    url = os.environ.get("OLLAMA_URL")
    if url:
        try:
            import requests
            r = requests.post(url.rstrip("/") + "/api/generate", json={
                "model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
                "prompt": ("You are the analyst inside a food-price forecasting app. "
                           "Answer ONLY from these facts; if the facts don't cover it, say so "
                           "plainly. Be short and simple.\n\nFACTS:\n" + ctx +
                           "\n\nQUESTION: " + question),
                "stream": False}, timeout=120)
            r.raise_for_status()
            return {"answer": r.json().get("response", "").strip(), "brain": "ollama"}
        except Exception as e:
            note = f"(Ollama not reachable: {str(e)[:80]} - using built-in brain.)\n\n"
            return {"answer": note + _rules(question, state, ctx), "brain": "rules"}
    return {"answer": _rules(question, state, ctx), "brain": "rules"}


def _rules(q: str, state: dict, ctx: str) -> str:
    ql = q.lower()
    ev, fc = state.get("eval"), state.get("forecast")
    if any(w in ql for w in ("accura", "score", "good", "trust", "how well", "perform")):
        if ev is None or not len(ev):
            return "Nothing is trained yet. Upload data and press Train."
        wins = int(ev["model_wins"].sum())
        return (f"Honest answer: in {len(ev)} fair exams (train on past, test on the next "
                f"6 months), the model beat the simple baseline in {wins} of {len(ev)}. "
                "When it dared say 'up' or 'down', it was right "
                f"{ev['right_when_it_did'].iloc[-1]} of the time in the latest exam. "
                "Food prices are noisy - treat forecasts as a guide, not a promise.")
    if any(w in ql for w in ("why", "clue", "feature", "driver", "important")):
        imp = state.get("importances")
        if imp is None:
            return "Train first, then I can show which clues the model used."
        return ("The clues that helped most: " +
                ", ".join(imp.head(5).index) +
                ". ('vs_nat' = this market compared to the national price; 'c1/c3/c12' = "
                "recent price changes; 'ms/mc' = time of year.)")
    if any(w in ql for w in ("up", "rise", "increase", "expensive", "down", "fall", "drop", "forecast", "next month")):
        if fc is None or not len(fc):
            return "No forecast yet - press Train first."
        up = fc[fc["trend"] == "up"].nlargest(5, "forecast")
        dn = fc[fc["trend"] == "down"].nsmallest(5, "forecast")
        lines = [f"Forecast for {fc['month'].iloc[0]:%B %Y}:"]
        if len(up):
            lines.append("Likely UP: " + "; ".join(f"{r.commodity} in {r.market}" for r in up.itertuples()))
        if len(dn):
            lines.append("Likely DOWN: " + "; ".join(f"{r.commodity} in {r.market}" for r in dn.itertuples()))
        lines.append("Check the forecast table for every market, with low-high ranges.")
        return "\n".join(lines)
    if any(w in ql for w in ("shock", "anomal", "alarm", "strange", "weird")):
        a = state.get("anomalies")
        if a is None or not len(a):
            return "No shocks computed yet - press Train first."
        return ("Recent shocks (price jumped more than 3x the market's normal wobble): " +
                "; ".join(f"{r.commodity} in {r.market} {r.month:%Y-%m} ({r.change_pct:+.0f}%)"
                          for r in a.head(6).itertuples()))
    if any(w in ql for w in ("llm", "ollama", "brain", "ai model")):
        return ("I answer with simple rules grounded in the trained results. For free-form "
                "answers, install Ollama on this computer (ollama.com), run "
                "`ollama pull llama3.2`, then start this app with OLLAMA_URL=http://localhost:11434")
    return ("I can answer about: accuracy, why (clues), forecast up/down, shocks. "
            "Here is everything I currently know:\n\n" + ctx)
