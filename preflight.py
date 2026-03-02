"""Pre-flight checks before the expensive LLM run."""
import sys, os, json, hashlib, time
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# ── Load .env ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(override=True)  # override=True needed: ANTHROPIC_API_KEY has empty system env var

SEP = "=" * 72

# ── 1. API KEY SMOKE TEST ────────────────────────────────────────────────
print(f"\n{SEP}")
print("1. API KEY SMOKE TEST — one call per provider")
print(SEP)

test_prompt = "Reply with exactly: OK"

# OpenAI
print("\n  [GPT-4o] ...", end=" ", flush=True)
try:
    from openai import OpenAI
    client = OpenAI(timeout=30)
    resp = client.chat.completions.create(
        model="gpt-4o", messages=[{"role":"user","content":test_prompt}], max_tokens=10)
    print(f"✓  ({resp.choices[0].message.content.strip()[:20]})")
except Exception as e:
    print(f"✗  {type(e).__name__}: {str(e)[:100]}")

# Gemini
print("  [Gemini 2.5 Flash] ...", end=" ", flush=True)
try:
    import warnings
    warnings.filterwarnings("ignore", message=".*google.generativeai.*")
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(test_prompt)
    print(f"✓  ({resp.text.strip()[:20]})")
except Exception as e:
    print(f"✗  {type(e).__name__}: {str(e)[:100]}")

# Claude — must pass api_key explicitly
print("  [Claude Sonnet 4] ...", end=" ", flush=True)
try:
    import anthropic
    aclient = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=30)
    resp = aclient.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10, messages=[{"role":"user","content":test_prompt}])
    print(f"✓  ({resp.content[0].text.strip()[:20]})")
except Exception as e:
    print(f"✗  {type(e).__name__}: {str(e)[:100]}")

# ── 2. CACHE HIT ANALYSIS ────────────────────────────────────────────────
print(f"\n{SEP}")
print("2. CACHE HIT ANALYSIS")
print(SEP)

CACHE_FILE = Path("cache/response_cache.json")
cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
print(f"  Cache entries: {len(cache):,}")

# Load sample to compute expected cache keys
markets_df = pd.read_parquet("cache/markets.parquet")

COMP_MIN, COMP_MAX = 0.10, 0.90
MODELS = {
    "gpt":    {"name": "GPT-4o",           "knowledge_cutoff": date(2023, 10, 1)},
    "gemini": {"name": "Gemini 2.5 Flash", "knowledge_cutoff": date(2025, 1, 1)},
    "claude": {"name": "Claude Sonnet 4",  "knowledge_cutoff": date(2025, 3, 1)},
}

eligible = markets_df[
    markets_df["kalshi_prob"].between(COMP_MIN, COMP_MAX) &
    markets_df["outcome"].notna() &
    markets_df["rules_primary"].str.len().gt(30)
].copy()

# Must match notebook's ck function exactly (colon separator)
def ck(*args):
    """16-char SHA-256 cache key."""
    return hashlib.sha256(":".join(str(a) for a in args).encode()).hexdigest()[:16]

total_calls = len(eligible) * len(MODELS)
hits = 0
misses_by_model = {mk: 0 for mk in MODELS}
hits_by_model = {mk: 0 for mk in MODELS}

for mk in MODELS:
    for ticker in eligible["ticker"]:
        key = ck("t1v2", mk, ticker)
        if key in cache:
            hits_by_model[mk] += 1
            hits += 1
        else:
            misses_by_model[mk] += 1

new_calls = total_calls - hits
print(f"  Total LLM calls needed: {total_calls:,}")
print(f"  Cache hits:             {hits:,}  ← will be instant")
print(f"  New calls needed:       {new_calls:,}  ← these cost money")
print()
print(f"  Per-model breakdown:")
for mk, cfg in MODELS.items():
    h = hits_by_model[mk]
    m = misses_by_model[mk]
    print(f"    {cfg['name']:22s}: {h:,} cached + {m:,} new = {h+m:,} total")

# ── 3. COST ESTIMATE ─────────────────────────────────────────────────────
print(f"\n{SEP}")
print("3. COST & TIME ESTIMATE")
print(SEP)
# Rough per-call costs (input + output tokens for a ~500 token prompt + ~300 token response)
cost_per_call = {"gpt": 0.008, "gemini": 0.001, "claude": 0.008}
total_cost = sum(misses_by_model[mk] * cost_per_call[mk] for mk in MODELS)
# Time: 4 workers, ~1.5s per call average (API latency)
est_seconds = new_calls * 1.5 / 4
est_minutes = est_seconds / 60
print(f"  Estimated cost:  ~${total_cost:.2f}")
print(f"  Estimated time:  ~{est_minutes:.0f} minutes ({est_minutes/60:.1f} hours)")
print(f"  Workers: 4 | ~{4/1.5:.1f} calls/sec effective throughput")

# ── 4. PROMPT CHECK — new "between" NO-outcome markets ───────────────────
print(f"\n{SEP}")
print("4. PROMPT CHECK — 'between' NO-outcome markets (newly included)")
print(SEP)

between_no = eligible[(eligible["direction"] == "between") & (eligible["outcome"] == 0)]
print(f"  'between' + outcome=NO: {len(between_no):,} markets")
print(f"  (Previously excluded by actual_temp_f.notna() filter)")
print()

for i, (_, row) in enumerate(between_no.head(3).iterrows()):
    print(f"  Example {i+1}: {row['ticker']}")
    print(f"    Title: {row['title'][:80]}")
    print(f"    Rules: {str(row['rules_primary'])[:120]}...")
    print(f"    Outcome: {int(row['outcome'])}  actual_temp_f: {row.get('actual_temp_f', 'N/A')}")
    print()

# ── 5. T-1 PRICE COVERAGE ────────────────────────────────────────────────
print(f"\n{SEP}")
print("5. T-1 PRICE COVERAGE")
print(SEP)
t1_df = pd.read_parquet("cache/kalshi_t1_prices.parquet")
t1_in_sample = t1_df[t1_df["ticker"].isin(eligible["ticker"])]
has_t1 = t1_in_sample["kalshi_t1_prob"].notna().sum()
total_sample = len(eligible)
print(f"  Markets with T-1 price: {has_t1}/{total_sample} ({has_t1/total_sample:.1%})")
print(f"  Markets without T-1:    {total_sample - has_t1}")
no_t1 = t1_in_sample[t1_in_sample["kalshi_t1_prob"].isna()]
if len(no_t1):
    print(f"  No-T1 reason: {no_t1['t1_error'].value_counts().to_dict()}")

# ── 6. SAMPLE INTEGRITY ──────────────────────────────────────────────────
print(f"\n{SEP}")
print("6. SAMPLE INTEGRITY")
print(SEP)

earliest_cutoff = min(cfg["knowledge_cutoff"] for cfg in MODELS.values())
latest_cutoff = max(cfg["knowledge_cutoff"] for cfg in MODELS.values())

def cutoff_period(event_date):
    d = pd.Timestamp(event_date).date()
    if d < earliest_cutoff: return "pre_all"
    if d >= latest_cutoff: return "post_all"
    return "transition"

eligible["cutoff_period"] = eligible["event_date"].apply(cutoff_period)

print(f"  Total markets:  {len(eligible):,}")
print(f"  YES rate:       {eligible['outcome'].mean():.1%}")
print()
for p in ["pre_all", "transition", "post_all"]:
    sub = eligible[eligible["cutoff_period"] == p]
    if len(sub):
        print(f"  {p:12s}: N={len(sub):>5}  YES={sub['outcome'].mean():.1%}  "
              f"NO={1-sub['outcome'].mean():.1%}")
print()
for mk, cfg in MODELS.items():
    post = (pd.to_datetime(eligible["event_date"]).dt.date >= cfg["knowledge_cutoff"]).sum()
    pre = len(eligible) - post
    print(f"  {cfg['name']:22s}: pre={pre:>5}  post={post:>5}")

print(f"\n{SEP}")
print("PRE-FLIGHT COMPLETE")
print(SEP)
if new_calls > 0:
    print(f"\n  ✓ Ready to run {new_calls:,} new LLM calls (~${total_cost:.2f}, ~{est_minutes:.0f} min)")
    print(f"  ✓ {hits:,} cached responses will be reused instantly")
else:
    print(f"\n  ✓ ALL {total_calls:,} calls are cached — no new API calls needed!")
