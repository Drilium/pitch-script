# Pitch Script

Premier League matchup and player-prop analyzer. The soccer companion to
[game-script](https://github.com/Drilium/game-script).

Every player prop comes with the chance of going over each line, a fair price in
American odds, the start chance behind it, and the matchup that moves it. Every
"defense allows" number opens the exact players and games behind it.

## Pages

| Page | What's on it |
|---|---|
| **Matchday** | This gameweek's fixtures: expected goals, win/draw/loss, over 2.5, BTTS, clean sheets. Strongest prop numbers of the week by market. |
| **Match** | Match markets with fair odds (and bookmaker odds when connected), player props for both teams, injuries, set-piece takers, team comparison, what each defense gives up by position (clickable), recent meetings. |
| **Player** | Next-match props for every line ("if he starts" and "if he plays"), how the number is built, a sportsbook line checker, season totals and per-90 for both seasons, history vs the opponent, every match. Fantasy (FPL) info sits in its own section at the bottom. |
| **Props** | Every projected player for any market and line, filterable. Line checker watchlist sorted by edge (saved in your browser). |
| **Players / Teams / Compare** | Search and sort, team pages with squad and schedule, referee foul and card rates, side-by-side player comparison. |
| **Model & data** | Backtest results, calibration chart, this season's out-of-sample grading, every data source and its status, data checks, and the gaps free data can't fill. |

## How the projections work

1. **Team strength.** Attack and defense ratings from expected goals (75%) and goals
   (25%), adjusted for opponent, with last season counted as 12 matches of evidence.
   Promoted clubs start at last season's bottom-three level.
2. **Match goals.** Poisson score grid with a Dixon-Coles low-score correction.
   Totals are kept 70% league average because, in the 2025-26 backtest, team
   strength had almost no relationship with how many goals a match had.
3. **Minutes.** Start chance and minutes when starting, weighted toward recent
   selection, multiplied by FPL's injury status.
4. **Player rates.** Per-90 rates shrunk toward the player's own last season (at 60%
   weight) and then the position average, so short hot streaks don't swing it.
5. **Matchup.** Attacking stats scale with the opponent's defensive rating and
   home/away. Tackles, fouls, cards and saves scale with what the opponent gives up
   to that position.
6. **Probabilities.** Negative binomial (Poisson with extra spread fitted per stat
   from real data), then a calibration fitted on last season's backtest.

## Accuracy

Walk-forward backtest on 2025-26, GW6–38: before each gameweek the model saw only
matches already played. Graded on players who started, against their raw hit rate
in earlier starts (what most prop apps show). Brier score, lower is better.

| Market | Lines graded | Model | Raw hit rate | Improvement |
|---|---|---|---|---|
| Anytime goal | 6,546 | 0.086 | 0.092 | +7.3% |
| Assist | 6,546 | 0.088 | 0.095 | +7.1% |
| Tackles over 1.5 | 6,546 | 0.217 | 0.231 | +6.2% |
| Keeper saves over 2.5 | 660 | 0.244 | 0.266 | +8.1% |
| Carded | 7,206 | 0.126 | 0.135 | +6.8% |
| Win / draw / loss | 330 matches | 0.623 | 0.654 (league rates) | |
| Over 2.5 goals | 330 matches | 0.249 | 0.247 (league rate) | none |

The model beats hit rates on every player market and on match winners, but not on
goal totals. This season (GW3–6, never used for tuning) the calibrated model is
11–13% better than hit rates on goals, tackles and cards.

## Data

All free. Details, health checks and the paid options are in
[DATA_SOURCES.md](DATA_SOURCES.md).

- **FPL official API**: minutes, goals, assists, xG, xA, tackles, saves, cards, injuries, set pieces (Opta-sourced)
- **Understat**: shots, shots on target, key passes
- **ESPN public JSON**: fouls committed and drawn
- **football-data.co.uk**: team shots, fouls, corners, cards, referees, bookmaker odds

## Setup

1. Push this folder to a new GitHub repo.
2. **Settings → Actions → General → Workflow permissions**: choose *Read and write*.
3. **Actions → Refresh Premier League data → Run workflow** once. The first run pulls
   last season's Understat and ESPN data, caches it in `cache/`, and re-runs the
   backtest with shots and fouls included.
4. Connect the repo to Netlify (no build command, publish directory `.`). Each data
   commit redeploys the site.

The workflow runs daily at 07:17 UTC (3:17am ET), so weekend, Monday and midweek
results land the next morning.

## Run locally

```bash
pip install -r requirements.txt
python build_data.py              # refresh data and projections
python build_data.py --backtest   # also re-run last season's backtest (~1 min)
python -m http.server             # open http://localhost:8000
```

## Files

| File | Purpose |
|---|---|
| `build_data.py` | Pulls every source, runs checks, builds projections, writes `soccer_data.json` |
| `sources.py` | One fetcher per source, with fallbacks and team/player name matching |
| `model.py` | Team ratings, minutes, player rates, matchup factors, probabilities, calibration |
| `backtest.py` | Walk-forward grading against hit-rate baselines |
| `index.html` | The site (no framework, no build step) |
| `cache/` | Last season's external data and backtest, so daily runs stay fast |

Portfolio project. Not betting advice.
