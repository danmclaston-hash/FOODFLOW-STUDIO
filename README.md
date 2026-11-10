# FoodFlow Studio

Feed it food-price datasets → it checks them like a doctor, trains a
regression model with honest leak-free testing, forecasts next month's
prices with a low–high range, flags price shocks, and answers questions.

## Run it (3 steps)

```bash
pip install -r requirements.txt
uvicorn app:app --port 8000
# open http://localhost:8000 in your browser
```

**Install as an app (PWA):** open it in Chrome → menu → "Install FoodFlow
Studio". Works on desktop and Android. On a phone, run the server on your
PC and open `http://<your-pc-ip>:8000` on the same Wi-Fi.

**Check it works:** upload `wfp_food_prices_uga.csv` (or any WFP country
file from https://data.humdata.org/dataset/wfp-food-prices), press
*Check my data*, then *Train & examine*. You should see the exam table
and a forecast within about a minute.

## What the app does, in plain words

1. **Data doctor** – finds your date/market/food/price columns even with
   different names, fixes units and price types, drops impossible prices,
   reports every fix, and asks you the questions a careful analyst would.
2. **Extra clues** – upload more CSVs (weather, fuel, exchange rate, news
   counts) with a `month` column (+ optional `market`). They are joined
   automatically and lagged by one month so the model can never peek at
   the future.
3. **Fair exams** – walk-forward testing: train on the past only, predict
   the next 6 months, slide forward, 4 times. Always compared to the lazy
   guess "same as last month". If the model can't beat that, the app says so.
4. **Forecast** – next month per food per market, with a low–high range
   and an up/down/steady arrow. Download as CSV.
5. **Unsupervised extras** – K-means market groups and a shock alarm
   (months that moved > 3× that market's normal wobble).
6. **Analyst** – answers questions from the trained results only.

## Free open-source LLM (optional)

Install [Ollama](https://ollama.com) on your PC, then:

```bash
ollama pull llama3.2
OLLAMA_URL=http://localhost:11434 uvicorn app:app --port 8000
```

The analyst then answers free-form questions, grounded in your real
numbers. Without Ollama a built-in rule brain answers (never invents).

## Live data connectors (run on your PC – they need internet)

- **Satellite weather (NASA POWER, free, no account):**
  `python connectors/nasa_power.py your_price_file.csv` → `weather.csv`
  → upload it in the app → retrain.
- **News / instability (GDELT via Google BigQuery, free tier):**
  `python connectors/gdelt_news.py` prints the exact steps and SQL.
- **Fresh prices:** re-download your country's CSV from WFP on HDX
  monthly and re-upload. (CHIRPS v3 rain can be added the same way later.)

## Honest limits

- It predicts **price changes**, which stand in for distribution stress.
  It shows patterns; it does not prove cause.
- Food prices are noisy. Expect the model to beat the baseline by a few
  percent, take a side ~40% of the time, and be right ~60% when it does.
- No synthetic data anywhere: if a file is unusable, the doctor says so.
- This is a local, single-user tool. To put it on the internet for a team
  (VPS + login), that is a next phase.

## Folder map

```
app.py                  the web server (FastAPI)
engine/pipeline.py      cleaning, features, models, exams, forecasts
engine/doctor.py        the data doctor
engine/analyst.py       the question answerer (+ Ollama hook)
static/                 the PWA (page, manifest, service worker, icons)
connectors/             NASA POWER + GDELT helpers
workspace/              your uploaded files and forecast.csv land here
```

## Hosting: how the pieces fit

```
your PC  ──(train)──▶  FoodFlow Studio (this app, Python)
                           │  publishes results with the SERVICE key
                           ▼
                     Supabase (Postgres)  ◀── reads with the public ANON key ──  Cloudflare Pages
                     runs · forecasts · shocks                                   /dashboard (static)
```

- **GitHub** holds the code. Pushing to `main` redeploys the dashboard.
- **Supabase** project `foodflow` (eu-west-1) holds results. Tables: `runs`,
  `forecasts`, `shocks`. Row Level Security: anyone can read, only the
  service key can write. Schema is in the project's migration `foodflow_schema`.
- **Cloudflare Pages** hosts `dashboard/` (see `dashboard/README.md`).
- **Why not run the ML on Cloudflare?** Workers can't run scikit-learn
  training. The brain stays on your PC (or a small VPS later); only results go online.

### Enable publishing from the app
```bash
export SUPABASE_URL=https://fdrzjxyhxwwkordscicn.supabase.co
export SUPABASE_SERVICE_KEY=<Supabase -> Project Settings -> API -> service_role>
uvicorn app:app --port 8000
```
After every *Train & examine*, results are pushed as a new run and the
dashboard shows the latest one. Never commit the service key.
