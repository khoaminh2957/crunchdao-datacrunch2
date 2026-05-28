# CrunchDAO Setup Walkthrough

End-to-end: zero account → first submission ready. ~30 minutes total.

## 1. Sign up (5 min)

1. Open https://hub.crunchdao.com/
2. Click **Sign Up**. Use your real email (`vnden20051111@gmail.com`).
3. Verify email (check inbox + spam).
4. Complete profile: display name, optional bio. Display name appears on public leaderboard — pick something you're OK with showing publicly.

## 2. Install Python deps (3 min)

```powershell
cd C:\Users\Admin\earn5usd\crunchdao
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify CLI installed:

```powershell
crunch --version
```

Should print something like `crunch-cli 5.x.x`.

## 3. Get API key (2 min)

1. Logged into hub.crunchdao.com → click your avatar → **Settings** → **API Keys**.
2. Click **Generate New Key**. Copy immediately — it shows once.
3. Save to `.env` in this directory:

```powershell
"CRUNCHDAO_API_KEY=ck_xxxxxxxxxxxxxxxxxx" | Out-File -FilePath .env -Encoding utf8
```

(Replace `ck_xxx...` with your actual key.)

## 4. Choose active competition (5 min)

List active competitions (note: v11 CLI uses `list`, NOT `competitions`):

```powershell
crunch list
```

Recommended priorities (verified on hub 2026-05-24):

1. **datacrunch** — continuous timeseries tournament (active, fastest feedback).
2. **adialab** — ADIA Lab market-forecasting timeseries (higher prize pool).
3. **causality-discovery** — DAG/causal track, longer cycle.
4. **structural-break-real-time** — break-detection unstructured track.

The full list at time of writing also includes: broad-obesity-1/2/3, numinous, btcdvol, synth, datacrunch-2, xentiment, structural-break*, falcon, broad-1/2/3, mid-one, datacrunch-rally, venture-capital-portfolio-prediction, pi.

Pick one, then initialize the project. **The `--token` flag is REQUIRED in CLI v11** — there is no way to run `setup` without one. It accepts either your API key (`ck_xxx`) OR a per-project "clone token" generated from the hub UI for that competition.

```powershell
crunch setup <competition-name> <your-project-name> --token <ck_or_clone_token>
```

Example:

```powershell
crunch setup datacrunch my-first-sub --token ck_xxxxxxxxxxxxxxxxxx
```

`crunch list` and `crunch ping` work WITHOUT a token (useful for confirming the CLI is wired up before you have credentials).

This creates `my-first-sub/` with starter `main.py`, `data/`, and config. **Do not delete** — the rest of this pipeline writes into that directory.

## 5. Wire wallet for USDC payout (5 min)

1. Install MetaMask: https://metamask.io/download/
2. Create wallet, **save seed phrase offline** (NOT in cloud, NOT in screenshot).
3. Add Polygon network: MetaMask → Networks → Add network → search "Polygon Mainnet" → Add.
4. Copy your wallet address (starts with `0x...`).
5. CrunchDAO hub → **Settings** → **Wallet** → paste address → Save.

> Without a wallet linked, earnings accrue but you cannot withdraw. Do this step BEFORE first submission.

## 6. First sanity check (5 min)

```powershell
cd my-first-sub
python ..\download_data.py
python ..\train_baseline.py
python ..\submit.py --dry-run
```

`--dry-run` runs `crunch test` (local validation) without pushing. Confirms:
- Model trains without error
- Output schema matches competition spec
- Prediction file size sane

If green, drop `--dry-run` to push live:

```powershell
python ..\submit.py
```

## 7. Track rank (2 min)

After ~30 min (depending on competition eval cadence), your submission appears on the leaderboard:

```powershell
python ..\leaderboard_track.py
```

Appends your current rank + score to `leaderboard_history.csv`. Run via `cron_daily.ps1` to log a daily trace.

## Common gotchas

| Symptom | Fix |
|---------|-----|
| `crunch: command not found` | Activate venv: `.\.venv\Scripts\Activate.ps1` |
| `401 Unauthorized` on push | API key wrong or expired — regenerate in hub Settings |
| `Invalid prediction schema` | Run `crunch test` first — it prints expected columns |
| Submission ranked very low | Normal for baseline. Iterate features, not hyperparams |
| Payout shows 0 after 30 days | Check that competition ROUND is closed; some pay end-of-round only |

## Next steps after first submission

1. Read the competition's data documentation (in `data/` after first `crunch download`).
2. Improve features — your WQ-style ts_zscore/ts_backfill patterns translate well.
3. Add ensemble diversity (LightGBM + CatBoost + neural net), not just LGBM tuning.
4. Submit at least 2x per week to stay on the live leaderboard.
