# Public dashboard (Cloudflare Pages)

A static page that reads the latest run from Supabase (read-only anon key).

Deploy, option A (no code, recommended):
  Cloudflare dashboard -> Workers & Pages -> Create -> Pages -> Connect to Git
  -> pick the FoodFlow repo -> Build command: (empty) -> Build output directory: dashboard
  Every git push then redeploys automatically.

Deploy, option B (one command, needs Node):
  npx wrangler pages deploy dashboard --project-name foodflow

The anon key is PUBLIC by design: Row Level Security only lets it read.
Writes need the service key, which stays on your PC (see engine/publish.py).
