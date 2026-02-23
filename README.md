# Forecasting Showdown: Prediction Markets vs. Frontier LLMs

Compares forecasting accuracy across three categories:

| Category | Forecasters |
|----------|-------------|
| Prediction markets | Polymarket, Kalshi |
| Frontier LLMs — vanilla | GPT, Gemini, Claude (training data only) |
| Frontier LLMs — tool-augmented | Same models + real-time FRED, EIA & Open-Meteo data |

Performance is measured with **Brier scores** and **hypothetical betting returns** against market prices.

### Domains
- **Federal Funds Rate** — Will the Fed cut at upcoming 2026 FOMC meetings?
- **US Retail Gas Prices** — Will national average exceed/fall below price thresholds?
- **City Temperature** — Will the average daily high in [city] exceed its 10-year historical median in [month] 2026?

All temperature resolution months (March–June 2026) are post-training-cutoff for every model tested, ensuring genuine probabilistic forecasting rather than recall.

---

## Quick Start

### Prerequisites
- Python 3.10+
- API keys (see below)

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/forecasting-showdown-llms-vs-markets.git
cd forecasting-showdown-llms-vs-markets
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

| Key | Where to get it | Cost |
|-----|----------------|------|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Paid |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Free tier available |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Paid |
| `FRED_API_KEY` | [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) | Free |
| `EIA_API_KEY` | [eia.gov/opendata](https://www.eia.gov/opendata/) | Free |

> Polymarket, Kalshi, and **Open-Meteo** (temperature data) require no authentication or API key.

### 3. Run the notebook

```bash
jupyter notebook lab.ipynb
```

Run cells top-to-bottom. API responses are cached in `cache/` so subsequent runs are fast and cheap.

---

## Project Structure

```
.
├── lab.ipynb          # Main notebook — all code and analysis
├── requirements.txt   # Python dependencies
├── .env.example       # API key template (copy to .env)
├── .gitignore
└── cache/             # Auto-created; stores cached API responses
```

## Sections

| Section | Description |
|---------|-------------|
| 1.1 — FRED | Federal funds rate (daily) |
| 1.2 — EIA | US weekly retail gasoline prices |
| 1.3 — Polymarket | Prediction market probabilities |
| 1.4 — Kalshi | Prediction market probabilities |
| 1.5 — Temperature | 10-year climate normals + Kalshi weather markets (Open-Meteo, no key) |
| 2 — Question Design | Binary forecasting questions with resolution criteria |
| 3 — LLM Forecasting | Vanilla and tool-augmented forecasting across three models |
| 4 — Scoring | Brier scores and hypothetical returns |
| 5 — Visualizations | Calibration plots, heatmaps, cumulative returns |
| 6 — Discussion | Summary statistics and interpretation |

## Temperature Domain Details

Six US cities: **New York, Chicago, Miami, Los Angeles, Denver, Seattle**

Four forecast months: **March, April, May, June 2026**

Each question asks whether the average daily high will exceed the **10-year historical median** for that city and month. Using the historical median as threshold sets the expected base rate at ≈ 50%, which maximises statistical power. Temperature data comes from [Open-Meteo](https://open-meteo.com/) (ERA5 reanalysis) — no API key required.

## Notes

- Questions resolve throughout 2026 — re-run the notebook after each resolution date to see updated Brier scores.
- The `cache/` directory stores all API responses as JSON; delete it to force fresh calls.
- The tool-augmented LLMs can call `get_temperature_data(city)` to retrieve recent temps and historical normals before forecasting.
