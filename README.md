# CrunchDAO Submission Pipeline

**Target:** First $5 USDC payout in 4–6 weeks if model is competitive.
**Stake:** None. Pure ML competition — no downside, only upside.
**Payout:** Monthly USDC to your Ethereum/Polygon wallet (MetaMask or Phantom EVM).

---

## What CrunchDAO is

CrunchDAO runs **data science competitions** funded by two main sources:

1. **DataCrunch** — In-house quant tournament (cross-sectional equity returns, similar to Numerai). Weekly/monthly forecasting rounds.
2. **ADIA Lab** — Sponsored by Abu Dhabi Investment Authority's research lab. Hosts academic-style competitions (causal discovery, market forecasting, structural break detection). Higher prize pools, longer cycles (8–16 weeks).

Both pay out in **USDC**. No staking, no NMR equivalent. You submit predictions, you get ranked, top participants share the pool.

## How payout works

- Each competition publishes a **prize pool** (typically $5k–$50k per round for ADIA, smaller continuous pool for DataCrunch).
- Pool is distributed across **top 200–300 ranks** on the final leaderboard.
- Payout is roughly geometric-decay: rank 1 takes ~5–10% of pool, rank 100 takes ~$5–$20, rank 300 takes the minimum threshold (often $1–$5).
- Wallet payout happens **monthly** for active competitions, **end-of-competition** for ADIA Lab rounds.

## Realistic first-payout expectation

- **Best case (top quant background, e.g. yours):** Rank ~150–250 on DataCrunch within 3–4 submission cycles → $5–$15 USDC after ~4 weeks.
- **Median case:** Baseline LightGBM ensemble lands rank 400–600 → no payout. Need 2–3 iteration cycles to crack top 300.
- **Worst case:** Submission has data leakage or wrong format → disqualified, 0 USDC.

**Hard truth:** $5 in 7 days is NOT realistic. CrunchDAO competitions evaluate on out-of-sample data that ships weekly/monthly. Even a perfect model on submission day won't crystallize a payout until the live evaluation window closes (typically 2–4 weeks per round).

## Why your background helps

- **FinDPO / sentiment alpha work** → directly relevant to ADIA Lab "market forecasting" tracks.
- **WQ Brain operator fluency** → translates to feature engineering on tabular financial data.
- **LightGBM/CV discipline** → DataCrunch baseline is essentially a cleaner version of what you already do for cluster1/cluster2 alpha generation.

The main gap is **leaderboard awareness** — these competitions reward marginal-edge feature engineering and ensemble diversity, not raw model size. Plan to spend 80% of cycles on feature selection, 20% on model tuning.

## Pipeline contents

| File | Purpose |
|------|---------|
| `setup.md` | Step-by-step signup, CLI install, API key wiring |
| `requirements.txt` | Python deps (crunch-cli + sklearn stack) |
| `download_data.py` | Pulls active competition data via CLI |
| `train_baseline.py` | LightGBM + Ridge ensemble with time-series CV |
| `submit.py` | Local validation (`crunch test`) then push (`crunch push`) |
| `cron_daily.ps1` | Daily refresh-and-resubmit if new data ships |
| `leaderboard_track.py` | Logs your daily rank to CSV for trend analysis |

## Recommended starting competition

**ADIA Lab Market Forecasting** (when active) — overlaps most with your existing skillset and has larger prize pools ($20k+). Fallback: **DataCrunch** continuous tournament for faster feedback loops.

## Wallet setup

You need an **EVM-compatible wallet** to receive USDC:

- **MetaMask** (recommended) — browser extension, supports Ethereum mainnet + Polygon. CrunchDAO pays on **Polygon** to avoid gas fees.
- **Phantom** (EVM mode) — also works if you already use it for Solana.

Add your wallet address in CrunchDAO dashboard → Settings → Payout. Without this, your earnings accumulate but cannot be withdrawn.

---

**Bottom line:** This is the cleanest no-stake ML earnings track for someone with your background. But it is a *competition*, not a faucet. Budget 4–6 weeks and 3–5 submission iterations before expecting the first USDC to land.
