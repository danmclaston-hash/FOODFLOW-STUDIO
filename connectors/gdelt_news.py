"""News connector (GDELT 2.0) - instability signals per country per month.

GDELT reads the world's news every 15 minutes and turns it into numbers.
We take two monthly numbers per country:
  - unrest_events: count of protest/conflict events reported
  - news_tone:     average tone of the news (negative = bad news)
Upload the resulting CSV as an extra table (joined on month).

HOW TO GET IT (free, but needs a Google account):
1. Go to https://console.cloud.google.com -> create a project (free tier:
   1 TB of BigQuery queries per month, far more than this needs).
2. Open BigQuery and run the query below (change the country code:
   Uganda = UG), then export the result as CSV.
3. Rename columns to: month, unrest_events, news_tone  -> upload in the app.

-- BigQuery SQL (copy from here) -------------------------------------------
SELECT
  FORMAT_DATE('%Y-%m-01', PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))) AS month,
  COUNTIF(EventRootCode IN ('14','17','18','19','20')) AS unrest_events,
  AVG(AvgTone) AS news_tone
FROM `gdelt-bq.gdeltv2.events`
WHERE ActionGeo_CountryCode = 'UG'
  AND SQLDATE >= 20060101
GROUP BY month
ORDER BY month;
-- -------------------------------------------------------------------------

Why not automatic? BigQuery needs your own Google login, which only you
can do. Once you have the CSV, the app does the rest.

Honest note: news counts partly measure how much journalists LOOK at a
place, not only what happens there. Treat them as a clue, not truth.
"""
print(__doc__)
