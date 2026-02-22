# Forecasting Showdown: Prediction Markets vs. Frontier LLMs

Compares forecasting accuracy across three categories:

| Category | Forecasters |
|----------|-------------|
| Prediction markets | Polymarket, Kalshi |
| Frontier LLMs — vanilla | GPT, Gemini, Claude (training data only) |
| Frontier LLMs — tool-augmented | Same models + real-time FRED & EIA data |

Performance is measured with **Brier scores** and **hypothetical betting returns** against market prices.

### Domains
- **Federal Funds Rate** — Will the Fed cut at upcoming 2026 FOMC meetings?
- **US Retail Gas Prices** — Will national average exceed/fall below price thresholds?

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

> Polymarket and Kalshi data require no authentication.

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
| 1 — Data Collection | FRED, EIA, Polymarket, Kalshi APIs |
| 2 — Question Design | Binary forecasting questions with resolution criteria |
| 3 — LLM Forecasting | Vanilla and tool-augmented forecasting across three models |
| 4 — Scoring | Brier scores and hypothetical returns |
| 5 — Visualizations | Calibration plots, heatmaps, cumulative returns |
| 6 — Discussion | Summary statistics and interpretation |

## Notes

- Many questions resolve throughout 2026 — re-run the notebook after resolution dates to see final Brier scores.
- The `cache/` directory stores API responses as JSON; delete it to force fresh API calls.
- The tool-augmented LLM section makes live API calls to FRED and EIA during inference (also cached).
