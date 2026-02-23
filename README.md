# Forecasting Showdown: Kalshi Weather Markets vs. Frontier LLMs

Can frontier LLMs beat liquid prediction markets at forecasting daily high temperatures?

This project evaluates three forecaster types against **8,000+ historical Kalshi binary weather markets** across six US cities, using Kalshi's own settlement records as ground truth.

---

## Experimental Design

| Forecaster | Prompt date | Information available |
|---|---|---|
| **AI @ T-7** | event_date − 7 days | Historical climatology only (no real-time data) |
| **AI @ T-1** | event_date − 1 day | Recent daily highs via Open-Meteo (`get_recent_weather` tool) |
| **Kalshi Market** | Final settlement price | Live crowd wisdom across all participants |

**Models tested**: GPT-4o, Gemini 2.5 Flash, Claude Sonnet 4.6

**Ground truth**: Kalshi's `expiration_value` field — the exact NWS-recorded temperature used to settle each contract.

**Metric**: Brier Score `BS = (1/N) Σ(pᵢ − oᵢ)²`
(0 = perfect · 0.25 = always-50% baseline · 1 = maximally wrong)

---

## Data

**Cities**: New York City, Chicago, Miami, Los Angeles, Denver, Seattle

**Markets**: All settled Kalshi daily high-temperature contracts (series `KXHIGHNY`, `KXHIGHCHI`, `KXHIGHMIA`, `KXHIGHLAX`, `KXHIGHDEN`, `KXHIGHTSEA`) — 8,000+ markets dating back to August 2021.

**Train / Validate / Test split** (chronological by event date):

| Split | Date range | Purpose |
|---|---|---|
| Train | before 2024-01-01 | Historical baseline |
| Validate | 2024-01-01 – 2025-08-31 | Within most LLM training windows |
| Test | 2025-09-01 onward | Post most LLM training cutoffs |

**Sample for AI evaluation**: 50 markets per split (highest-volume per city × calendar month) = **150 markets** × 3 models × 2 snapshots = ~900 AI calls, all cached after first run.

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

> **Open-Meteo** (recent weather data for T-1 tool calls) requires no API key.

### 3. Run the notebook

```bash
jupyter notebook lab.ipynb
```

Run cells top-to-bottom. All Kalshi market data and LLM responses are cached in `cache/` — subsequent runs are instant and free.

---

## Project Structure

```
final_project/
├── lab.ipynb          # Main notebook — all code and analysis
├── requirements.txt   # Python dependencies
├── .env               # API keys (gitignored — create manually)
├── .gitignore
└── cache/             # Auto-created on first run
    ├── markets.parquet    # All settled Kalshi markets (~50k rows)
    ├── response_cache.json  # LLM response cache (JSON, keyed by SHA-256)
    ├── results.parquet    # Master results DataFrame
    ├── results.csv        # Same, as CSV for easy inspection
    └── results_plots.png  # Generated charts
```

## Notebook Sections

| Section | Description |
|---|---|
| **1 — Market Data** | Fetch all settled Kalshi markets, descriptive stats, Kalshi's own Brier score |
| **2 — Sample Selection** | Stratified sample: highest-volume market per city × month per split |
| **3 — LLM Forecasting** | T-7 vanilla and T-1 tool-augmented forecasts across all three models |
| **4 — Brier Score Analysis** | Overall, by split, by city, generalisation gap (validate → test) |
| **5 — Visualisations** | Brier by model × snapshot, generalisation gap, AI vs market scatter, calibration curves |
| **6 — Summary** | Final leaderboard, best forecaster per city |

---

## Reloading Results

After running the notebook once, results persist and can be reloaded without re-running any API calls:

```python
import pandas as pd
results_df = pd.read_parquet("cache/results.parquet")
# or: pd.read_csv("cache/results.csv")
```

Key columns: `ticker`, `city`, `event_date`, `threshold_f`, `direction`, `actual_temp_f`, `kalshi_prob`, `outcome`, `result`, `snapshot`, `model`, `method`, `probability`, `brier`, `split`
