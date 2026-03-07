#!/usr/bin/env python3
"""frontier_models.py — Frontier Model Extension for Forecasting Showdown

Extends the LLM vs Kalshi weather forecasting study (lab.ipynb) by evaluating
five frontier AI models released or significantly upgraded post-2024:

  • o3              (OpenAI reasoning,      knowledge cutoff: Apr 2024)
  • Claude Opus 4.6 (Anthropic top-tier,   knowledge cutoff: Aug 2025)
  • Gemini 2.5 Pro  (Google top-tier,      knowledge cutoff: Jan 2025)
  • Grok 3          (xAI,                  knowledge cutoff: Jul 2025)
  • DeepSeek-R1     (DeepSeek reasoning,   knowledge cutoff: Jan 2025)

These are benchmarked against:
  • The original three models from lab.ipynb (GPT-4o, Gemini 2.5 Flash, Claude Sonnet 4)
  • The Kalshi T-1 market consensus
  • Always-50% and City-Month Rate baselines

Prerequisites:
  - lab.ipynb must have been run to populate:
      cache/markets.parquet
      cache/weather.parquet
      cache/kalshi_t1_prices.parquet
      cache/results.parquet   (for combined comparison)
  - .env must contain API keys for all desired providers
  - pip install openai anthropic google-generativeai python-dotenv pandas pyarrow
    numpy matplotlib seaborn tqdm

Usage:
  python frontier_models.py [--dry-run] [--models o3,grok3,...] [--workers N]

  --dry-run    Print cost/cache estimates without calling any LLM
  --models     Comma-separated subset of FRONTIER_MODELS keys to run
  --workers    ThreadPoolExecutor workers (default 4)
"""

import os, sys, re, json, hashlib, time, warnings, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dotenv import load_dotenv

# ── Setup ──────────────────────────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)

# ── Argument Parsing ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Frontier model extension for Forecasting Showdown")
parser.add_argument("--dry-run",  action="store_true", help="Estimate cost/cache without calling APIs")
parser.add_argument("--models",   type=str, default="", help="Comma-separated model keys to run (default: all)")
parser.add_argument("--workers",  type=int, default=4,  help="Parallel worker threads (default: 4)")
args = parser.parse_args()

SEP = "=" * 72

# ── Paths ──────────────────────────────────────────────────────────────────
CACHE_DIR        = Path("cache")
MARKETS_FILE     = CACHE_DIR / "markets.parquet"
WEATHER_FILE     = CACHE_DIR / "weather.parquet"
T1_PRICES_FILE   = CACHE_DIR / "kalshi_t1_prices.parquet"
EXISTING_RESULTS = CACHE_DIR / "results.parquet"     # from lab.ipynb
FRONTIER_CACHE   = CACHE_DIR / "frontier_response_cache.json"
FRONTIER_RESULTS = CACHE_DIR / "frontier_results.parquet"
FRONTIER_CSV     = CACHE_DIR / "frontier_results.csv"
COMBINED_RESULTS = CACHE_DIR / "combined_results.parquet"
FRONTIER_PLOTS   = CACHE_DIR / "frontier_results_plots.png"
PROGRESS_LOG     = CACHE_DIR / "frontier_run_progress.log"

for p in [CACHE_DIR, PROGRESS_LOG.parent]:
    p.mkdir(exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY",    "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY    = os.environ.get("GOOGLE_API_KEY",    "")
XAI_API_KEY       = os.environ.get("XAI_API_KEY",       "")
DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY",  "")

# ── Sample-selection parameters (must match lab.ipynb) ─────────────────────
COMP_MIN, COMP_MAX = 0.10, 0.90

# ── Original models from lab.ipynb (for combined comparison display) ────────
ORIGINAL_MODELS = {
    "gpt":    {"name": "GPT-4o",           "knowledge_cutoff": date(2023, 10, 1)},
    "gemini": {"name": "Gemini 2.5 Flash", "knowledge_cutoff": date(2025,  1, 1)},
    "claude": {"name": "Claude Sonnet 4",  "knowledge_cutoff": date(2025,  3, 1)},
}

# ── Frontier Model Registry ────────────────────────────────────────────────
FRONTIER_MODELS = {
    # ── Reasoning models ───────────────────────────────────────────────────
    "o3": {
        "name":             "o3",
        "provider":         "openai_reasoning",  # o-series: no temp param, max_completion_tokens
        "model_id":         "o3",
        "knowledge_cutoff": date(2024, 4, 1),
        "cost_per_call":    0.060,               # conservative: ~$60/M output tokens
        "notes":            "OpenAI reasoning model; uses max_completion_tokens, no temperature",
    },
    "deepseek_r1": {
        "name":             "DeepSeek-R1",
        "provider":         "deepseek",          # OpenAI-compatible via api.deepseek.com
        "model_id":         "deepseek-reasoner",
        "knowledge_cutoff": date(2025, 1, 1),
        "cost_per_call":    0.002,               # very cost-effective reasoning model
        "notes":            "DeepSeek reasoning model via OpenAI-compatible API",
    },
    # ── Top-tier chat models ───────────────────────────────────────────────
    "claude_opus": {
        "name":             "Claude Opus 4.6",
        "provider":         "anthropic",
        "model_id":         "claude-opus-4-6",
        "knowledge_cutoff": date(2025, 8, 1),
        "cost_per_call":    0.075,               # ~$75/M output tokens (top-tier)
        "notes":            "Anthropic's most capable model; highest knowledge cutoff",
    },
    "gemini_pro": {
        "name":             "Gemini 2.5 Pro",
        "provider":         "google",
        "model_id":         "gemini-2.5-pro",
        "knowledge_cutoff": date(2025, 1, 1),
        "cost_per_call":    0.015,
        "notes":            "Google's most capable Gemini model",
    },
    "grok3": {
        "name":             "Grok 3",
        "provider":         "xai",               # OpenAI-compatible via api.x.ai/v1
        "model_id":         "grok-3",
        "knowledge_cutoff": date(2025, 7, 1),
        "cost_per_call":    0.015,
        "notes":            "xAI's frontier model; OpenAI-compatible API",
    },
}

# ── Prompt templates (identical to lab.ipynb for fair comparison) ──────────
SYSTEM_PROMPT = (
    "You are an expert weather forecaster and superforecaster specialising in "
    "prediction market calibration. Your sole task is to estimate the probability "
    "that a specific Kalshi binary weather market resolves YES.\n\n"
    "Guidelines:\n"
    "- Use the provided recent temperature observations as your primary evidence.\n"
    "- Ground your estimate in seasonal climatology for the city and month.\n"
    "- Be well-calibrated: a 70% probability should resolve YES ~70% of the time.\n"
    "- Avoid anchoring to round numbers (0.25, 0.50, 0.75) unless clearly justified.\n\n"
    "End your response with EXACTLY one line:\n"
    "PROBABILITY: X.XX\n"
    "where X.XX is a decimal in [0.00, 1.00]."
)

USER_T1 = (
    "Today is {forecast_date} - the market resolves tomorrow.\n\n"
    "Market title: {title}\n\n"
    "Resolution criteria: {rules_primary}\n\n"
    "{weather_context}\n\n"
    "Based on the temperature observations above and your knowledge of "
    "seasonal patterns for {city} in {month_name}, "
    "estimate the probability this market resolves YES. "
    "Think step by step, then state your probability."
)


# ── Cache helpers ──────────────────────────────────────────────────────────
def ck(*args) -> str:
    """16-char SHA-256 cache key (same hash logic as lab.ipynb)."""
    return hashlib.sha256(":".join(str(a) for a in args).encode()).hexdigest()[:16]


def load_cache() -> dict:
    if FRONTIER_CACHE.exists():
        return json.loads(FRONTIER_CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    FRONTIER_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


CACHE_LOCK = Lock()


# ── Data loading ───────────────────────────────────────────────────────────
def load_data():
    """Load markets, weather, T-1 prices; build sample_df matching lab.ipynb."""
    for f in [MARKETS_FILE, WEATHER_FILE, T1_PRICES_FILE]:
        if not f.exists():
            sys.exit(f"ERROR: {f} not found — run lab.ipynb first to populate the cache.")

    markets_df  = pd.read_parquet(MARKETS_FILE)
    weather_df  = pd.read_parquet(WEATHER_FILE)
    t1_prices   = pd.read_parquet(T1_PRICES_FILE)

    # Apply same competitive-market filter as lab.ipynb
    sample_df = markets_df[
        markets_df["kalshi_prob"].between(COMP_MIN, COMP_MAX) &
        markets_df["outcome"].notna() &
        markets_df["rules_primary"].str.len().gt(30)
    ].copy()

    # Cutoff-period labeling (uses ALL models: original + frontier)
    all_cutoffs = (
        [cfg["knowledge_cutoff"] for cfg in ORIGINAL_MODELS.values()] +
        [cfg["knowledge_cutoff"] for cfg in FRONTIER_MODELS.values()]
    )
    earliest_cutoff = min(all_cutoffs)
    latest_cutoff   = max(all_cutoffs)

    def cutoff_period(event_date):
        d = pd.Timestamp(event_date).date()
        if d < earliest_cutoff: return "pre_all"
        if d >= latest_cutoff:  return "post_all"
        return "transition"

    sample_df["cutoff_period"] = sample_df["event_date"].apply(cutoff_period)

    # Per-frontier-model post-cutoff flag
    for mk, cfg in FRONTIER_MODELS.items():
        sample_df[f"post_cutoff_{mk}"] = (
            pd.to_datetime(sample_df["event_date"]).dt.date >= cfg["knowledge_cutoff"]
        )

    return sample_df, weather_df, t1_prices


# ── Weather context (identical to lab.ipynb) ───────────────────────────────
def get_weather_context(city: str, forecast_date: str, weather_df: pd.DataFrame,
                        days_back: int = 10) -> str:
    fd  = pd.Timestamp(forecast_date)
    cw  = weather_df[
        (weather_df["city"] == city) & (weather_df["date"] < fd)
    ].sort_values("date").tail(days_back)
    if len(cw) == 0:
        return f"(No weather history available for {city} before {forecast_date})"
    lines = [f"Observed daily high temperatures for {city} (degF):"]
    for _, row in cw.iterrows():
        if pd.notna(row["temp_f"]):
            lines.append(f"  {row['date'].strftime('%Y-%m-%d')}: {row['temp_f']:.1f}")
    return "\n".join(lines)


# ── Client initialization (lazy, so --dry-run skips API key checks) ────────
_clients_initialized = False
_oai_client = _ant_client = _gem_model = _xai_client = _deepseek_client = None


def _init_clients(active_models: set):
    """Initialize only the API clients needed for the active model set."""
    global _clients_initialized, _oai_client, _ant_client, _gem_model
    global _xai_client, _deepseek_client
    if _clients_initialized:
        return
    _clients_initialized = True

    providers_needed = {FRONTIER_MODELS[mk]["provider"] for mk in active_models}

    if "openai" in providers_needed or "openai_reasoning" in providers_needed:
        import openai as _oai_lib
        _oai_client = _oai_lib.OpenAI(api_key=OPENAI_API_KEY, timeout=120.0)

    if "anthropic" in providers_needed:
        import anthropic as _ant_lib
        _ant_client = _ant_lib.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=_ant_lib.Timeout(120.0, connect=10.0),
        )

    if "google" in providers_needed:
        import google.generativeai as _genai_lib
        _genai_lib.configure(api_key=GOOGLE_API_KEY)
        _gem_model = _genai_lib.GenerativeModel(
            FRONTIER_MODELS["gemini_pro"]["model_id"],
            system_instruction=SYSTEM_PROMPT,
        )

    if "xai" in providers_needed:
        import openai as _oai_lib
        _xai_client = _oai_lib.OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1",
            timeout=120.0,
        )

    if "deepseek" in providers_needed:
        import openai as _oai_lib
        _deepseek_client = _oai_lib.OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
            timeout=120.0,
        )


# ── Model dispatch ─────────────────────────────────────────────────────────
def call_model(model_key: str, user_content: str) -> str:
    """Call the appropriate API and return raw text output. Raises on error."""
    cfg      = FRONTIER_MODELS[model_key]
    provider = cfg["provider"]
    model_id = cfg["model_id"]

    if provider == "openai":
        resp = _oai_client.chat.completions.create(
            model=model_id, temperature=0, max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        return resp.choices[0].message.content

    if provider == "openai_reasoning":
        # o-series models: no temperature parameter, use max_completion_tokens
        resp = _oai_client.chat.completions.create(
            model=model_id,
            max_completion_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        return resp.choices[0].message.content

    if provider == "anthropic":
        resp = _ant_client.messages.create(
            model=model_id, max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.content[0].text

    if provider == "google":
        resp = _gem_model.generate_content(
            user_content,
            request_options={"timeout": 120},
        )
        return resp.text

    if provider == "xai":
        resp = _xai_client.chat.completions.create(
            model=model_id, temperature=0, max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        return resp.choices[0].message.content

    if provider == "deepseek":
        # DeepSeek-R1 (deepseek-reasoner): supports system messages
        # Note: reasoning content arrives in resp.choices[0].message.reasoning_content
        # but the final answer is in .content
        resp = _deepseek_client.chat.completions.create(
            model=model_id, max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unknown provider: {provider!r}")


# ── Probability parser (identical to lab.ipynb) ────────────────────────────
def parse_prob(text: str) -> float:
    text = str(text)
    m = re.search(r"PROBABILITY:\s*(0\.\d+|1\.0+|0\.0+)", text)
    if m:
        return float(m.group(1))
    # Fallback: last decimal-looking number in the tail of the response
    hits = re.findall(r"\b(0\.\d+|1\.0)\b", text[-500:])
    return float(hits[-1]) if hits else np.nan


# ── Progress logging ───────────────────────────────────────────────────────
def _log(msg: str, also_print: bool = False):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if also_print:
        print(line)


# ── Forecasting loop ───────────────────────────────────────────────────────
def run_frontier(sample_df: pd.DataFrame, weather_df: pd.DataFrame,
                 cache: dict, active_models: list, n_workers: int) -> pd.DataFrame:
    """Run all frontier model forecasts in parallel, with caching and retry."""

    tasks     = [(mk, row) for mk in active_models for _, row in sample_df.iterrows()]
    new_count = [0]

    _s = {
        "t0": time.time(), "done": 0,
        "hits": 0, "new": 0, "errors": 0, "parse_ok": 0, "parse_fail": 0,
        "by_model": {mk: {"hits": 0, "new": 0, "err": 0, "ok": 0}
                     for mk in active_models},
    }
    _first_errors: dict = {}
    SAVE_EVERY = 10
    LOG_EVERY  = 50

    def _snapshot():
        elapsed = time.time() - _s["t0"]
        rate = _s["done"] / elapsed if elapsed > 0 else 0
        eta  = (len(tasks) - _s["done"]) / rate if rate > 0 else float("inf")
        _log(
            f"done={_s['done']}/{len(tasks)}  hits={_s['hits']}  "
            f"new={_s['new']}  err={_s['errors']}  "
            f"parse_ok={_s['parse_ok']}  parse_fail={_s['parse_fail']}  "
            f"rate={rate:.1f}/s  ETA={eta/60:.1f}m",
            also_print=True,
        )

    def forecast_one(args):
        model_key, row = args
        cfg = FRONTIER_MODELS[model_key]
        # Use distinct cache-version prefix so frontier cache never collides with lab.ipynb
        key = ck("frontier_v1", model_key, row["ticker"])

        # ── Cache hit ─────────────────────────────────────────────────────
        with CACHE_LOCK:
            if key in cache:
                output = cache[key]["output"]
                prob   = parse_prob(output)
                _s["hits"] += 1
                _s["by_model"][model_key]["hits"] += 1
                if pd.notna(prob):
                    _s["parse_ok"] += 1
                    _s["by_model"][model_key]["ok"] += 1
                else:
                    _s["parse_fail"] += 1
                return {
                    "ticker": row["ticker"], "model": cfg["name"],
                    "model_key": model_key, "snapshot": "T-1",
                    "probability": prob, "raw_output": str(output),
                }, None

        # ── Build prompt ──────────────────────────────────────────────────
        fd  = (pd.Timestamp(row["event_date"]) - timedelta(days=1)).strftime("%Y-%m-%d")
        mon = pd.Timestamp(row["event_date"]).strftime("%B")
        wx  = get_weather_context(row["city"], fd, weather_df)

        user_content = USER_T1.format(
            forecast_date=fd, title=row["title"],
            rules_primary=row["rules_primary"],
            city=row["city"], month_name=mon, weather_context=wx,
        )

        # ── Call model with exponential backoff ───────────────────────────
        output = None
        for attempt in range(3):
            try:
                output = call_model(model_key, user_content)
                break
            except Exception as e:
                err = str(e)
                if model_key not in _first_errors:
                    _first_errors[model_key] = f"{type(e).__name__}: {err[:200]}"
                is_throttle = any(kw in err.lower() for kw in
                                  ["rate_limit", "429", "timeout", "timed out", "overload"])
                if attempt < 2 and is_throttle:
                    time.sleep(5 * (2 ** attempt))
                else:
                    output = f"ERROR: {e}"
                    break

        if output is None:
            output = "ERROR: max retries exceeded"

        is_error = str(output).startswith("ERROR:")
        prob     = parse_prob(output)

        # ── Update stats ──────────────────────────────────────────────────
        if is_error:
            _s["errors"] += 1
            _s["by_model"][model_key]["err"] += 1
        else:
            _s["new"] += 1
            _s["by_model"][model_key]["new"] += 1
        if pd.notna(prob):
            _s["parse_ok"] += 1
            _s["by_model"][model_key]["ok"] += 1
        else:
            _s["parse_fail"] += 1

        save_entry = None if is_error else (key, output)

        return {
            "ticker": row["ticker"], "model": cfg["name"],
            "model_key": model_key, "snapshot": "T-1",
            "probability": prob, "raw_output": str(output),
        }, save_entry

    _log(f"Starting frontier run: {len(tasks)} tasks, {n_workers} workers", also_print=True)
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(forecast_one, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Frontier forecasts"):
            result, update = future.result()
            results.append(result)
            _s["done"] += 1
            if update:
                k, out = update
                with CACHE_LOCK:
                    cache[k] = {"output": out}
                    new_count[0] += 1
                    if new_count[0] % SAVE_EVERY == 0:
                        save_cache(cache)
            if _s["done"] % LOG_EVERY == 0:
                _snapshot()

    save_cache(cache)
    _log(f"Run complete. {_s['new']} new calls, {_s['hits']} cache hits, "
         f"{_s['errors']} errors, {_s['parse_fail']} parse failures.", also_print=True)

    if _first_errors:
        print("\nFirst error per model:")
        for mk, msg in _first_errors.items():
            print(f"  {mk}: {msg}")

    return pd.DataFrame(results)


# ── Result assembly ────────────────────────────────────────────────────────
def assemble_results(t1_results: pd.DataFrame, sample_df: pd.DataFrame,
                     t1_prices: pd.DataFrame) -> pd.DataFrame:
    """Merge forecasts with market metadata; add Brier scores and baselines."""

    META_COLS = [
        "ticker", "city", "event_date", "direction", "threshold_f", "title",
        "rules_primary", "result", "outcome", "actual_temp_f", "kalshi_prob",
        "volume", "cutoff_period",
    ] + [f"post_cutoff_{mk}" for mk in FRONTIER_MODELS]

    meta_cols_present = [c for c in META_COLS if c in sample_df.columns]
    df = t1_results.merge(sample_df[meta_cols_present], on="ticker", how="left")

    # Brier score for each frontier forecast
    df["brier"] = np.where(
        df[["probability", "outcome"]].notna().all(axis=1),
        (df["probability"] - df["outcome"]) ** 2,
        np.nan,
    )

    # Per-row post_cutoff flag (for the specific model that made the forecast)
    df["post_cutoff"] = df.apply(
        lambda r: bool(r.get(f"post_cutoff_{r['model_key']}", False))
                  if r["model_key"] in FRONTIER_MODELS else False,
        axis=1,
    )

    # ── Baselines (computed on full sample, same logic as lab.ipynb) ───────
    pre_data = sample_df[sample_df["cutoff_period"] == "pre_all"].dropna(subset=["outcome"]).copy()
    pre_data["month"] = pd.to_datetime(pre_data["event_date"]).dt.month
    base_rate_map = {}
    for (city, month), g in pre_data.groupby(["city", "month"]):
        base_rate_map[(city, month)] = g["outcome"].mean() if len(g) >= 3 else sample_df["outcome"].mean()
    global_yes_rate = sample_df["outcome"].dropna().mean()

    def get_base_rate(city, event_date):
        month = pd.Timestamp(event_date).month
        return base_rate_map.get((city, month), global_yes_rate)

    baseline_rows = []
    for _, row in sample_df.dropna(subset=["outcome"]).iterrows():
        p50    = 0.50
        p_base = get_base_rate(row["city"], row["event_date"])
        for model_name, prob in [("Baseline: Always-50%", p50),
                                  ("Baseline: City-Month Rate", p_base)]:
            baseline_rows.append({
                **{c: row[c] for c in meta_cols_present if c in row.index},
                "model": model_name, "model_key": "baseline",
                "snapshot": "T-1", "probability": prob, "raw_output": "",
                "brier": (prob - row["outcome"]) ** 2, "post_cutoff": False,
            })

    # ── Kalshi T-1 rows ───────────────────────────────────────────────────
    t1_map = t1_prices.set_index("ticker")["kalshi_t1_prob"].to_dict()
    kalshi_rows = []
    for _, row in sample_df.dropna(subset=["outcome"]).iterrows():
        prob = t1_map.get(row["ticker"], np.nan)
        if pd.isna(prob):
            continue
        kalshi_rows.append({
            **{c: row[c] for c in meta_cols_present if c in row.index},
            "model": "Kalshi T-1", "model_key": "kalshi_t1",
            "snapshot": "T-1", "probability": prob, "raw_output": "",
            "brier": (prob - row["outcome"]) ** 2, "post_cutoff": False,
        })

    results_df = pd.concat(
        [df, pd.DataFrame(baseline_rows), pd.DataFrame(kalshi_rows)],
        ignore_index=True,
    )

    results_df.to_parquet(FRONTIER_RESULTS, index=False)
    results_df.to_csv(FRONTIER_CSV, index=False)
    print(f"\nSaved: {FRONTIER_RESULTS}  ({len(results_df):,} rows)")

    # ── Merge with original lab.ipynb results if available ─────────────────
    if EXISTING_RESULTS.exists():
        orig = pd.read_parquet(EXISTING_RESULTS)
        # Exclude original baselines/Kalshi rows — we recomputed them above
        orig_ai = orig[orig["model_key"].isin(ORIGINAL_MODELS.keys())].copy()
        combined = pd.concat([orig_ai, results_df], ignore_index=True)
        combined.to_parquet(COMBINED_RESULTS, index=False)
        print(f"Saved: {COMBINED_RESULTS}  ({len(combined):,} rows, includes original models)")
        return combined
    else:
        print("(lab.ipynb results not found — combined comparison will use frontier models only)")
        return results_df


# ── Analysis ───────────────────────────────────────────────────────────────
def print_analysis(results_df: pd.DataFrame):
    """Print leaderboard, cutoff-effect table, and per-city breakdown."""
    scored = results_df.dropna(subset=["brier"]).copy()

    print(f"\n{SEP}")
    print("OVERALL BRIER SCORE LEADERBOARD  (lower = better; 0.25 = always-50% baseline)")
    print(SEP)

    overall = (
        scored.groupby("model")["brier"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "Mean Brier", "std": "Std", "count": "N"})
        .round(4)
        .sort_values("Mean Brier")
    )
    print(overall.to_string())

    # Highlight frontier vs original
    print(f"\n{SEP}")
    print("FRONTIER vs ORIGINAL MODELS (Frontier = 🆕)")
    print(SEP)
    frontier_names = {cfg["name"] for cfg in FRONTIER_MODELS.values()}
    for model_name, row in overall.iterrows():
        tag  = "🆕" if model_name in frontier_names else "  "
        diff = row["Mean Brier"] - overall.loc["Kalshi T-1", "Mean Brier"] if "Kalshi T-1" in overall.index else float("nan")
        verdict = "BEATS market" if diff < -0.005 else ("matches market" if abs(diff) <= 0.005 else "trails market")
        print(f"  {tag} {model_name:28s}  Brier={row['Mean Brier']:.4f}  N={int(row['N']):>5}  "
              f"vs Kalshi T-1: {diff:+.4f}  [{verdict}]")

    # ── Knowledge-cutoff effect ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("KNOWLEDGE-CUTOFF EFFECT (frontier models only)")
    print(SEP)

    frontier_keys = list(FRONTIER_MODELS.keys())
    ai_scored = scored[scored["model_key"].isin(frontier_keys)]
    if len(ai_scored) == 0:
        print("  No frontier model results to analyse.")
        return

    for mk, cfg in FRONTIER_MODELS.items():
        grp  = ai_scored[ai_scored["model_key"] == mk]
        pre  = grp.loc[~grp["post_cutoff"], "brier"]
        post = grp.loc[ grp["post_cutoff"], "brier"]
        if len(pre) == 0 and len(post) == 0:
            continue
        pre_str  = f"Brier={pre.mean():.4f}  N={len(pre):>5}" if len(pre) else "N/A"
        post_str = f"Brier={post.mean():.4f}  N={len(post):>5}" if len(post) else "N/A"
        delta    = post.mean() - pre.mean() if (len(pre) and len(post)) else float("nan")
        print(f"  {cfg['name']:22s}  pre={pre_str}  post={post_str}  Δ={delta:+.4f}")

    # ── By city ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("BY CITY (frontier models + Kalshi T-1 only)")
    print(SEP)

    city_models = frontier_keys + ["kalshi_t1"]
    city_scored = scored[scored["model_key"].isin(city_models)]
    if len(city_scored):
        city_table = (
            city_scored.groupby(["city", "model"])["brier"]
            .mean().unstack("model").round(4)
        )
        print(city_table.to_string())


# ── Visualisation ──────────────────────────────────────────────────────────
def plot_results(results_df: pd.DataFrame):
    """6-panel figure: leaderboard, cutoff effect, by city, calibration, direction."""
    scored = results_df.dropna(subset=["brier", "probability"]).copy()
    if len(scored) == 0:
        print("Nothing to plot — no scored rows.")
        return

    frontier_names = {cfg["name"] for cfg in FRONTIER_MODELS.values()}
    orig_names     = {cfg["name"] for cfg in ORIGINAL_MODELS.values()}
    all_ai_names   = frontier_names | orig_names

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Frontier Models vs Kalshi Markets — Forecasting Showdown Extension",
                 fontsize=14, fontweight="bold")
    palette = sns.color_palette("husl", n_colors=len(scored["model"].unique()))

    # ── Panel 1: Overall Brier leaderboard ───────────────────────────────
    ax = axes[0, 0]
    overall = (
        scored.groupby("model")["brier"]
        .mean().sort_values().reset_index()
        .rename(columns={"brier": "Mean Brier"})
    )
    colors = [
        "#e74c3c" if m in frontier_names else
        "#3498db" if m in orig_names else
        "#95a5a6"
        for m in overall["model"]
    ]
    bars = ax.barh(overall["model"], overall["Mean Brier"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0.25, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Always-50% baseline")
    ax.set_xlabel("Mean Brier Score (lower = better)")
    ax.set_title("Overall Brier Score Leaderboard")
    ax.legend(fontsize=8)
    ax.set_xlim(0, max(0.28, overall["Mean Brier"].max() + 0.01))
    # Value labels
    for bar, val in zip(bars, overall["Mean Brier"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7.5)
    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#e74c3c", label="Frontier (new)"),
                       Patch(facecolor="#3498db", label="Original (lab.ipynb)"),
                       Patch(facecolor="#95a5a6", label="Baseline / Market")]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

    # ── Panel 2: Pre vs post knowledge cutoff (frontier models) ─────────
    ax = axes[0, 1]
    frontier_scored = scored[scored["model_key"].isin(FRONTIER_MODELS.keys()) &
                             scored["post_cutoff"].notna()]
    if len(frontier_scored) > 0:
        cutoff_agg = (
            frontier_scored.groupby(["model", "post_cutoff"])["brier"]
            .mean().unstack("post_cutoff").rename(columns={False: "Pre-cutoff", True: "Post-cutoff"})
        )
        cutoff_agg.plot(kind="bar", ax=ax, color=["#2ecc71", "#e67e22"],
                        edgecolor="white", linewidth=0.5)
        ax.set_xlabel("")
        ax.set_ylabel("Mean Brier Score")
        ax.set_title("Pre vs Post Knowledge Cutoff\n(frontier models only)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.axhline(0.25, color="black", linestyle="--", linewidth=1, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No frontier model data", ha="center", va="center")
        ax.set_title("Pre vs Post Knowledge Cutoff")

    # ── Panel 3: By city (frontier models + Kalshi T-1) ──────────────────
    ax = axes[0, 2]
    city_models = list(FRONTIER_MODELS.keys()) + ["kalshi_t1"]
    city_scored = scored[scored["model_key"].isin(city_models)]
    if len(city_scored) > 0:
        city_agg = (
            city_scored.groupby(["city", "model"])["brier"]
            .mean().unstack("model")
        )
        city_agg.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("City")
        ax.set_ylabel("Mean Brier Score")
        ax.set_title("Performance by City\n(frontier models + Kalshi T-1)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        ax.axhline(0.25, color="black", linestyle="--", linewidth=1, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No city data", ha="center", va="center")
        ax.set_title("Performance by City")

    # ── Panel 4: Calibration curves ───────────────────────────────────────
    ax = axes[1, 0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    plot_models = [
        m for m in scored["model"].unique()
        if scored.loc[scored["model"] == m, "model_key"].iloc[0] in list(FRONTIER_MODELS.keys()) + ["kalshi_t1"]
    ]
    bin_edges = np.linspace(0, 1, 11)
    for i, model_name in enumerate(sorted(plot_models)):
        sub = scored[scored["model"] == model_name].dropna(subset=["probability", "outcome"])
        if len(sub) < 20:
            continue
        sub["bin"] = pd.cut(sub["probability"], bins=bin_edges, labels=False, include_lowest=True)
        cal = sub.groupby("bin").agg(mean_pred=("probability", "mean"),
                                      mean_outcome=("outcome", "mean"),
                                      count=("outcome", "count"))
        cal = cal[cal["count"] >= 5]
        ls = "--" if model_name == "Kalshi T-1" else "-"
        ax.plot(cal["mean_pred"], cal["mean_outcome"], marker="o", markersize=4,
                linestyle=ls, label=model_name, linewidth=1.5)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed YES rate")
    ax.set_title("Calibration Curves")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ── Panel 5: Market direction breakdown ───────────────────────────────
    ax = axes[1, 1]
    direction_models = list(FRONTIER_MODELS.keys()) + ["kalshi_t1"]
    dir_scored = scored[scored["model_key"].isin(direction_models) &
                        scored["direction"].notna()]
    if len(dir_scored) > 0:
        dir_agg = (
            dir_scored.groupby(["direction", "model"])["brier"]
            .mean().unstack("model")
        )
        dir_agg.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Market Direction")
        ax.set_ylabel("Mean Brier Score")
        ax.set_title("Performance by Market Direction\n(greater / less / between)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
        ax.legend(fontsize=6)
        ax.axhline(0.25, color="black", linestyle="--", linewidth=1, alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No direction data", ha="center", va="center")
        ax.set_title("Performance by Market Direction")

    # ── Panel 6: Brier score distribution (violin) ────────────────────────
    ax = axes[1, 2]
    violin_models = list(FRONTIER_MODELS.keys()) + ["kalshi_t1", "baseline"]
    violin_scored = scored[scored["model_key"].isin(violin_models)].copy()
    if len(violin_scored) > 0:
        model_order = (
            violin_scored.groupby("model")["brier"].mean()
            .sort_values().index.tolist()
        )
        sns.violinplot(data=violin_scored, x="brier", y="model", order=model_order,
                       ax=ax, orient="h", inner="quartile",
                       palette=["#e74c3c" if m in frontier_names else "#95a5a6"
                                for m in model_order])
        ax.axvline(0.25, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_xlabel("Brier Score")
        ax.set_ylabel("")
        ax.set_title("Brier Score Distribution\n(violin; quartiles shown)")
    else:
        ax.text(0.5, 0.5, "No violin data", ha="center", va="center")
        ax.set_title("Brier Score Distribution")

    plt.tight_layout()
    fig.savefig(FRONTIER_PLOTS, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots: {FRONTIER_PLOTS}")


# ── Preflight summary ──────────────────────────────────────────────────────
def print_preflight(sample_df: pd.DataFrame, cache: dict, active_models: list):
    """Print cost/cache estimates and API key status without making any calls."""
    print(f"\n{SEP}")
    print("FRONTIER MODELS PREFLIGHT CHECK")
    print(SEP)

    print(f"\n  Models to run: {', '.join(active_models)}")
    print(f"  Sample size:   {len(sample_df):,} markets")

    # API key check
    key_map = {
        "openai":           ("OPENAI_API_KEY",    OPENAI_API_KEY),
        "openai_reasoning": ("OPENAI_API_KEY",    OPENAI_API_KEY),
        "anthropic":        ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
        "google":           ("GOOGLE_API_KEY",    GOOGLE_API_KEY),
        "xai":              ("XAI_API_KEY",       XAI_API_KEY),
        "deepseek":         ("DEEPSEEK_API_KEY",  DEEPSEEK_API_KEY),
    }
    print("\n  API key status:")
    needed_providers = {FRONTIER_MODELS[mk]["provider"] for mk in active_models}
    for prov in sorted(needed_providers):
        key_name, key_val = key_map[prov]
        status = "✓ set" if key_val else "✗ MISSING"
        print(f"    {prov:20s}  {key_name}: {status}")

    # Cache analysis
    total_calls = len(sample_df) * len(active_models)
    hits        = sum(1 for mk in active_models for ticker in sample_df["ticker"]
                      if ck("frontier_v1", mk, ticker) in cache)
    new_calls   = total_calls - hits

    print(f"\n  Cache analysis:")
    print(f"    Total calls needed: {total_calls:,}")
    print(f"    Cache hits:         {hits:,}")
    print(f"    New calls needed:   {new_calls:,}")

    # Cost estimate
    total_cost = sum(
        sum(1 for ticker in sample_df["ticker"]
            if ck("frontier_v1", mk, ticker) not in cache)
        * FRONTIER_MODELS[mk]["cost_per_call"]
        for mk in active_models
    )
    est_seconds = new_calls * 2.0 / args.workers  # ~2s/call average for reasoning models
    print(f"\n  Cost estimate:")
    print(f"    Estimated cost: ~${total_cost:.2f}")
    print(f"    Estimated time: ~{est_seconds/60:.0f} minutes ({args.workers} workers)")

    # Per-model breakdown
    print("\n  Per-model breakdown:")
    for mk in active_models:
        cfg  = FRONTIER_MODELS[mk]
        h    = sum(1 for ticker in sample_df["ticker"] if ck("frontier_v1", mk, ticker) in cache)
        m    = len(sample_df) - h
        cost = m * cfg["cost_per_call"]
        print(f"    {cfg['name']:22s}: {h:,} cached + {m:,} new  ~${cost:.2f}  ({cfg['notes']})")

    print(f"\n{SEP}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"\n{SEP}")
    print("FRONTIER MODELS — Forecasting Showdown Extension")
    print(SEP)

    # Select models to run
    if args.models:
        requested = [m.strip() for m in args.models.split(",")]
        invalid   = [m for m in requested if m not in FRONTIER_MODELS]
        if invalid:
            sys.exit(f"ERROR: Unknown model keys: {invalid}. "
                     f"Valid keys: {list(FRONTIER_MODELS.keys())}")
        active_models = requested
    else:
        active_models = list(FRONTIER_MODELS.keys())

    print(f"\nFrontier models selected: {', '.join(active_models)}")

    # Load data
    print("\nLoading cached market data...")
    sample_df, weather_df, t1_prices = load_data()
    print(f"  Sample: {len(sample_df):,} competitive markets  "
          f"(YES rate: {sample_df['outcome'].mean():.1%})")

    cutoff_counts = sample_df["cutoff_period"].value_counts()
    for p in ["pre_all", "transition", "post_all"]:
        n = cutoff_counts.get(p, 0)
        print(f"    {p:12s}: N={n:>5}")

    # Load cache
    cache = load_cache()
    print(f"\nFrontier response cache: {len(cache):,} entries")

    # Preflight / dry-run
    print_preflight(sample_df, cache, active_models)

    if args.dry_run:
        print("\n[--dry-run] Stopping before any API calls.")
        return

    # Initialize clients for the active providers
    _init_clients(set(active_models))

    # Run forecasts
    print(f"\nStarting forecasts... (tail -f {PROGRESS_LOG} for live progress)")
    t1_results = run_frontier(sample_df, weather_df, cache, active_models, args.workers)

    # Assemble and score
    print("\nAssembling results...")
    results_df = assemble_results(t1_results, sample_df, t1_prices)

    # Print analysis
    print_analysis(results_df)

    # Visualize
    print("\nGenerating plots...")
    plot_results(results_df)

    print(f"\n{SEP}")
    print("DONE")
    print(f"  Results:  {FRONTIER_RESULTS}")
    print(f"  Combined: {COMBINED_RESULTS if EXISTING_RESULTS.exists() else '(lab.ipynb results not found)'}")
    print(f"  Plots:    {FRONTIER_PLOTS}")
    print(f"  Log:      {PROGRESS_LOG}")
    print(SEP)


if __name__ == "__main__":
    main()
