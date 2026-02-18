# Football Predictions (EPL, La Liga, UCL)

This project builds 1/X/2 predictions using API-Football data (fixtures, standings, injuries, lineups when available), with odds prior blending and base-rate calibration. It also supports backtesting-based calibration.

## Quick start
1. Create `config.yaml` from `config.example.yaml`.
2. Set your API key in env:

```bash
export API_FOOTBALL_KEY="YOUR_KEY"
```

3. Install deps:

```bash
pip install -r requirements.txt
```

4. Run lookup to confirm league IDs:

```bash
python -m src.main lookup-league --name "Premier League" --country "England"
```

5. Predict today:

```bash
python -m src.main predict --date today
```

## Notes
- Lineups are often available only close to kickoff; the model uses them if present.
- Injuries are used as a small penalty; you can tune weights in `config.yaml`.
- Odds blending is controlled by `weights.odds_prior` and `odds.enabled` in `config.yaml`.
- Base-rate calibration is controlled by `weights.calibration_blend`.
- Lineup position weights can be tuned via `weights.lineup_position_weights`.

## Backtest + Calibration
Run backtesting over prior seasons to build league-specific calibration:

```bash
python -m src.main backtest --seasons 3
```

This writes `data/calibration.json`, which is used automatically if `calibration.enabled: true`. Calibration is applied by strength buckets (home_strong / even / away_strong).

## Deploy (Public)
This project is ready for public deploy with `gunicorn`.

### Render
1. Push the project to GitHub.
2. In Render create a new Web Service from the repo.
3. Render can auto-detect settings from `render.yaml`, or set manually:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn -w ${WEB_CONCURRENCY:-2} -k gthread --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-120} -b 0.0.0.0:$PORT src.web:app`
4. Add environment variable:
- `API_FOOTBALL_KEY=<your_key>`

### Railway
1. Create a new project from the repo.
2. Set start command to the same `gunicorn` command above.
3. Add env var `API_FOOTBALL_KEY`.

### Local development
Use `run.sh`:

```bash
PORT=5001 ./run.sh
```
