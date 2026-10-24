# Data sources and gaps

Pitch Script runs on free data first. This file lists what each source provides,
what it can't, and what paying for would add.

## Free sources in the pipeline

| Source | What it gives | Used for | Notes |
|---|---|---|---|
| **FPL official API** (`fantasy.premierleague.com/api`) | Per player, per match: minutes, starts, goals, assists, xG, xA, xG conceded, tackles, clearances/blocks/interceptions, recoveries, defensive contributions, saves, cards, bonus. Plus injury status, chance of playing, news text, set-piece order, fixtures. | Every player rate, the minutes model, availability, team ratings (xG), fantasy section | Opta-sourced. Official and free. |
| **vaastav/Fantasy-Premier-League** (GitHub) | Same fields as above, mirrored as CSV | Fallback when the FPL API is unreachable; last two seasons for priors and the backtest | Lags the live API by about a week. |
| **Understat** (`understat.com`) | Per player, per match: shots, key passes, xG chain/buildup. Shot-level data (result of every shot) | Shots and shots-on-target props (SoT = goals + saved shots) | Unofficial JSON endpoints. Parsed two ways in case the site changes format. |
| **ESPN public JSON** (`site.api.espn.com`) | Per player, per match: fouls committed, fouls drawn, offsides, shots, shots on target, saves. Team possession | Fouls props; backup shots source | Undocumented and unlicensed. Could change without notice. |
| **football-data.co.uk** | Per match: shots, shots on target, fouls, corners, cards per team, referee, and bookmaker odds (Pinnacle, Bet365, market average, closing lines). `fixtures.csv` has this week's odds | Team comparison, referee table, market vs model on match odds | Free CSVs, updated twice a week. |
| **datasets/football-datasets** (GitHub) | football-data.co.uk results and team stats, no odds | Fallback for past seasons | |

### Health checks run on every build

- Player goals plus opponent own goals add up to every final score.
- Exactly 11 starters per team per match.
- football-data.co.uk scores match FPL scores (two independent feeds).
- Shots on target never exceed shots; goals never exceed shots on target.

All passed on the 2025-26 season (380/380 matches) and 2026-27 so far.

## What free data can't cover

| Gap | Why it matters | Option | Cost (checked Oct 2026) |
|---|---|---|---|
| **Passes per player** | Pass-attempt props for midfielders and center backs | **API-Football** `/fixtures/players`: passes (total, key, accuracy), tackles, fouls committed/drawn, shots, cards per player | Free 100 requests/day; **Pro $19/mo** (7,500/day); Ultra $29; Mega $39 |
| **Confirmed lineups** | Start chance becomes certain about an hour out | API-Football `/fixtures/lineups` (posted 20–40 min before kickoff) | Same plan |
| **Sportsbook player prop lines** | Automatic edge finding instead of typing lines in | **The Odds API**: EPL `player_shots_on_target` confirmed; player props mostly from US books, other soccer prop markets not confirmed on their docs | Free 500 credits/mo; 20K credits **$30/mo**; 100K $59/mo. Props are pulled one event at a time |
| **Predicted lineups** | Start chance before team news drops | **Sportmonks** Expected Lineups add-on | Base plan from €29/mo (5 leagues); add-on price not published |
| **Referee appointments** | Card and foul props move with the referee | No free API. The Premier League posts appointments midweek; would need a scraper | Free but fragile |
| **Event data** (progressive passes, touches in box, pressures, carries) | Deeper role and usage context | Opta (Stats Perform) or StatsBomb | Enterprise pricing. FBref lost its Opta feed in January 2026, so there's no free route |

### Recommendation

1. **API-Football Pro ($19/mo)** first. It fills passes, gives confirmed lineups,
   and is a licensed feed, so it can replace the ESPN dependency for fouls.
2. **The Odds API 20K ($30/mo)** second, once the projections are trusted, to turn
   the Props page's manual line checker into an automatic board. Verify which EPL
   prop markets and books it actually returns with the free tier before paying.
3. Skip event data. It's expensive and the prop markets that matter are already
   covered by the sources above.
