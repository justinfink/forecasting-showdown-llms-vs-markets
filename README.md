# Forecasting Showdown: LLMs vs Kalshi Weather Prediction Markets

Can frontier LLMs beat a liquid prediction market at forecasting daily high temperatures — at the **same information horizon**?

This project benchmarks three AI models against the **Kalshi T-1 market price** (the market's last pre-event consensus) across **1,500+ competitive binary weather markets** spanning six US cities, using Kalshi's own settlement records as ground truth.

---

## Experimental Design

| Forecaster | Snapshot | Information available |
|---|---|---|
| **GPT-4o** | T-1 (eve of event) | 10-day historical highs from Open-Meteo ERA5 archive |
| **Gemini 1.5 Flash** | T-1 | Same |
| **Claude 3.5 Sonnet** | T-1 | Same |
| **Kalshi T-1** | T-1 (last pre-midnight candle) | Live prediction market price |
| **Baseline: Always-50%** | — | None |
| **Baseline: City-Month Rate** | — | Historical YES rate for that city × month |

**Why T-1 for Kalshi?**  Kalshi weather markets close ~29 hours *after* the event date midnight, by which point the NWS official temperature is public knowledge.  The final `last_price` is not a pre-event forecast.  We use the **Kalshi T-1 price** — the last hourly candlestick that closed at or before event_date midnight UTC — fetched from Kalshi's historical candlestick API (`/series/{series}/markets/{ticker}/candlesticks`).  This gives both AI and market the same ~T-1 information horizon.

**Competitive markets only:**  ~94% of settled markets have `last_price` near 0 or 1 (trivially certain outcomes).  We restrict to markets where `last_price ∈ [0.10, 0.90]` — cases where the outcome was genuinely uncertain.

**Models chosen for knowledge-cutoff analysis:**

| Model | Knowledge Cutoff | Provider |
|---|---|---|
| GPT-4o | October 2023 | OpenAI |
| Gemini 1.5 Flash | November 2023 | Google |
| Claude 3.5 Sonnet (20241022) | April 2024 | Anthropic |

All three cutoffs fall within the Kalshi data window (Aug 2021 – present), providing substantial data both before and after each cutoff.

**Ground truth**: Kalshi's `expiration_value` — the exact NWS-recorded temperature used to settle each contract.

**Metric**: Brier Score `BS = (1/N) Σ(pᵢ − oᵢ)²`
(0 = perfect · 0.25 = always-50% baseline · 1 = maximally wrong)

---

## Data

**Cities**: New York City, Chicago, Miami, Los Angeles, Denver, Seattle

**Markets**: All settled Kalshi daily high-temperature contracts (series `KXHIGHNY`, `KXHIGHCHI`, `KXHIGHMIA`, `KXHIGHLAX`, `KXHIGHDEN`, `KXHIGHTSEA`) — 28,000+ markets, ~1,600 competitive.

**Knowledge-cutoff periods** (no train/test split — all markets used, with period labels):

| Period | Definition |
|---|---|
| `pre_all` | Before Oct 2023 — all models in training window |
| `transition` | Oct 2023 – Apr 2024 — models progressively reach cutoff |
| `post_all` | After Apr 2024 — all models past their training cutoff |

**LLM sample**: Up to 150 competitive markets per city = ~900 markets × 3 models = ~2,700 AI calls, all cached after first run.

---

## Quick Start

### Prerequisites
- Python 3.10+
- API keys (see below)

### 1. Clone and install

```bash
git clone https://github.com/justinfink/forecasting-showdown-llms-vs-markets.git
cd forecasting-showdown-llms-vs-markets/final_project
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set up API keys

Create a `.env` file in the `final_project/` directory:

```
KALSHI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

| Key | Where to get it | Cost |
|---|---|---|
| `KALSHI_API_KEY` | [kalshi.com](https://kalshi.com) → Account → API | Free |
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Paid |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Free tier available |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Paid |

> **Open-Meteo** (historical weather for prompt context) requires no API key.

### 3. Run the notebook

```bash
jupyter notebook lab.ipynb
```

Run cells top-to-bottom. All market data, Kalshi T-1 prices, and LLM responses are cached in `cache/` — subsequent runs are instant and free.

---

## Project Structure

```
final_project/
├── lab.ipynb                    # Main notebook — all code and analysis
├── requirements.txt             # Python dependencies
├── .env                         # API keys (gitignored — create manually)
├── .gitignore
└── cache/                       # Auto-created on first run
    ├── markets.parquet          # All settled Kalshi markets (~28k rows)
    ├── weather.parquet          # Open-Meteo ERA5 daily highs per city
    ├── kalshi_t1_prices.parquet # Pre-event candlestick prices (T-1)
    ├── response_cache.json      # LLM response cache (keyed by SHA-256)
    ├── results.parquet          # Master results DataFrame
    ├── results.csv              # Same, as CSV for easy inspection
    └── results_plots.png        # Generated charts
```

## Notebook Sections

| Section | Description |
|---|---|
| **1 — Market Data** | Fetch all settled Kalshi markets; confirm last_price is post-outcome |
| **2 — Sample Selection** | Competitive filter + cutoff-period labels; cap per city |
| **3 — Kalshi T-1 Prices** | Fetch genuine pre-event prices via candlestick API |
| **4 — LLM Forecasting** | T-1 forecasts with ERA5 weather context, all three models |
| **5 — Assemble Results** | Merge AI + Kalshi T-1 + baselines into single DataFrame |
| **6 — Brier Analysis** | Overall, AI vs market, by city, pre/post cutoff, temporal cohorts |
| **7 — Visualisations** | Brier bars, pre/post cutoff, cohort trends, calibration curves |
| **8 — Summary** | Final leaderboard, AI vs market verdict, cutoff-effect table |

---

## Reloading Results

After running the notebook once, results persist and can be reloaded without re-running any API calls:

```python
import pandas as pd
results_df = pd.read_parquet("cache/results.parquet")
# or: pd.read_csv("cache/results.csv")
```

Key columns: `ticker`, `city`, `event_date`, `direction`, `threshold_f`, `actual_temp_f`, `kalshi_prob`, `outcome`, `cutoff_period`, `model`, `model_key`, `probability`, `brier`, `post_cutoff`
