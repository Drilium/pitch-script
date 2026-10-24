"""
Data sources for Pitch Script (Premier League analyzer).

Every fetcher returns plain pandas DataFrames plus a status dict, so the
site can say exactly where each number came from and how fresh it is.

Free sources used
-----------------
1. FPL official API (fantasy.premierleague.com)   - Opta-sourced player match stats:
   minutes, starts, goals, assists, xG, xA, tackles, clearances/blocks/interceptions,
   recoveries, saves, cards, injury news, set-piece takers.
   Fallback: vaastav/Fantasy-Premier-League GitHub mirror (same fields, ~1 week lag).
2. Understat (understat.com)                       - per-player shots, shots on target
   (from shot-level data), key passes, xG chain.
3. ESPN public JSON (site.api.espn.com)            - per-player fouls committed / drawn,
   offsides, shots, shots on target; team possession. Undocumented, can change.
4. football-data.co.uk                             - team match stats (shots, SOT, fouls,
   corners, cards, referee) + bookmaker 1X2 / over-under odds, incl. upcoming fixtures.
   Fallback for past seasons: datasets/football-datasets GitHub mirror (no odds).
"""
from __future__ import annotations

import datetime as dt
import difflib
import io
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (pitch-script data pipeline; portfolio project)"}
TIMEOUT = 30

MIRROR_FPL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
MIRROR_FD = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/premier-league"
FPL_API = "https://fantasy.premierleague.com/api"
FD_BASE = "https://www.football-data.co.uk"
UNDERSTAT = "https://understat.com"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1"

POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def status(name, ok, rows=0, as_of=None, note="", url=""):
    return {"source": name, "ok": bool(ok), "rows": int(rows), "as_of": as_of, "note": note, "url": url}


def get(url, **kw):
    headers = dict(UA)
    headers.update(kw.pop("headers", {}))
    r = requests.get(url, headers=headers, timeout=kw.pop("timeout", TIMEOUT), **kw)
    r.raise_for_status()
    return r


def get_csv(url, **kw):
    return pd.read_csv(io.StringIO(get(url, **kw).content.decode("utf-8", "replace")), **{"low_memory": False})


def season_code(season):  # '2026-27' -> '2627'
    a, b = season.split("-")
    return a[2:] + b


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower().replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


TEAM_ALIASES = {
    "man united": "Man Utd", "manchester united": "Man Utd", "man utd": "Man Utd",
    "man city": "Man City", "manchester city": "Man City",
    "nott m forest": "Nott'm Forest", "nottm forest": "Nott'm Forest", "nottingham forest": "Nott'm Forest",
    "tottenham": "Spurs", "tottenham hotspur": "Spurs", "spurs": "Spurs",
    "newcastle": "Newcastle", "newcastle united": "Newcastle",
    "wolves": "Wolves", "wolverhampton wanderers": "Wolves", "wolverhampton": "Wolves",
    "west ham": "West Ham", "west ham united": "West Ham",
    "brighton": "Brighton", "brighton hove albion": "Brighton", "brighton and hove albion": "Brighton",
    "leeds": "Leeds", "leeds united": "Leeds",
    "sheffield united": "Sheffield Utd", "sheffield utd": "Sheffield Utd",
    "bournemouth": "Bournemouth", "afc bournemouth": "Bournemouth",
    "ipswich": "Ipswich Town", "ipswich town": "Ipswich Town",
    "hull": "Hull City", "hull city": "Hull City",
    "coventry": "Coventry City", "coventry city": "Coventry City",
    "leicester": "Leicester", "leicester city": "Leicester",
    "luton": "Luton", "luton town": "Luton",
    "sunderland": "Sunderland", "burnley": "Burnley", "everton": "Everton", "fulham": "Fulham",
    "chelsea": "Chelsea", "arsenal": "Arsenal", "liverpool": "Liverpool", "aston villa": "Aston Villa",
    "brentford": "Brentford", "crystal palace": "Crystal Palace", "southampton": "Southampton",
    "norwich": "Norwich", "norwich city": "Norwich", "watford": "Watford",
    "middlesbrough": "Middlesbrough", "west brom": "West Brom", "west bromwich albion": "West Brom",
}


def team_mapper(fpl_teams: pd.DataFrame):
    """Return f(name) -> FPL team id for one season's FPL teams table."""
    by_name = {norm_name(n): int(i) for n, i in zip(fpl_teams["name"], fpl_teams["id"])}
    by_short = {str(s).lower(): int(i) for s, i in zip(fpl_teams["short_name"], fpl_teams["id"])}

    def f(name):
        n = norm_name(name)
        if n in TEAM_ALIASES and norm_name(TEAM_ALIASES[n]) in by_name:
            return by_name[norm_name(TEAM_ALIASES[n])]
        if n in by_name:
            return by_name[n]
        if n in by_short:
            return by_short[n]
        m = difflib.get_close_matches(n, list(by_name), n=1, cutoff=0.6)
        return by_name[m[0]] if m else None

    return f


# ----------------------------------------------------------------------------
# 1. FPL (official API with GitHub-mirror fallback)
# ----------------------------------------------------------------------------
FPL_KEEP = [
    "element", "fixture", "opponent_team", "was_home", "kickoff_time", "round", "minutes", "starts",
    "goals_scored", "assists", "expected_goals", "expected_assists", "expected_goals_conceded",
    "goals_conceded", "clean_sheets", "saves", "penalties_saved", "penalties_missed", "own_goals",
    "yellow_cards", "red_cards", "tackles", "clearances_blocks_interceptions", "recoveries",
    "defensive_contribution", "bonus", "bps", "total_points", "influence", "creativity", "threat",
    "value", "team_h_score", "team_a_score",
]


def _fpl_official():
    boot = get(f"{FPL_API}/bootstrap-static/").json()
    fixtures = pd.DataFrame(get(f"{FPL_API}/fixtures/").json())
    players = pd.DataFrame(boot["elements"])
    teams = pd.DataFrame(boot["teams"])
    ids = players["id"].tolist()

    def hist(pid):
        for attempt in range(3):
            try:
                return get(f"{FPL_API}/element-summary/{pid}/").json().get("history", [])
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return []

    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = [r for h in ex.map(hist, ids) for r in h]
    gw = pd.DataFrame(rows)
    return players, teams, fixtures, gw


def _fpl_mirror(season):
    base = f"{MIRROR_FPL}/{season}"
    players = get_csv(f"{base}/players_raw.csv")
    teams = get_csv(f"{base}/teams.csv")
    fixtures = get_csv(f"{base}/fixtures.csv")
    gw = get_csv(f"{base}/gws/merged_gw.csv")
    return players, teams, fixtures, gw


def load_fpl(season, current=False):
    """Players, teams, fixtures and per-player-per-fixture rows for one season."""
    src, note = None, ""
    if current:
        try:
            players, teams, fixtures, gw = _fpl_official()
            src = "FPL official API"
        except Exception as e:  # blocked network, FPL downtime, off-season
            note = f"official API unavailable ({type(e).__name__}); used GitHub mirror"
    if src is None:
        players, teams, fixtures, gw = _fpl_mirror(season)
        src = "FPL via vaastav GitHub mirror"

    players = players[players["element_type"].isin([1, 2, 3, 4])].copy()
    fixtures = fixtures.copy()
    fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"], utc=True, errors="coerce")
    for c in ["team_h", "team_a", "event", "id"]:
        fixtures[c] = pd.to_numeric(fixtures[c], errors="coerce").astype("Int64")

    gw = gw[[c for c in FPL_KEEP if c in gw.columns]].copy()
    gw = gw[gw["element"].isin(players["id"])]
    for c in gw.columns:
        if c not in ("kickoff_time", "was_home"):
            gw[c] = pd.to_numeric(gw[c], errors="coerce")
    gw["was_home"] = gw["was_home"].astype(str).str.lower().isin(["true", "1"])
    # team for that match comes from the fixture (handles mid-season moves between PL clubs)
    fx = fixtures.set_index("id")[["team_h", "team_a", "event", "kickoff_time", "finished"]]
    gw = gw.join(fx, on="fixture", rsuffix="_fx")
    gw = gw.dropna(subset=["team_h", "team_a"])
    gw["team"] = np.where(gw["was_home"], gw["team_h"].astype(int), gw["team_a"].astype(int))
    gw["opp"] = np.where(gw["was_home"], gw["team_a"].astype(int), gw["team_h"].astype(int))
    gw["kickoff"] = gw["kickoff_time_fx"].fillna(pd.to_datetime(gw["kickoff_time"], utc=True, errors="coerce"))
    gw["gw"] = gw["event"].fillna(gw.get("round"))
    gw = gw.drop(columns=[c for c in ["team_h", "team_a", "event", "kickoff_time", "kickoff_time_fx", "round"] if c in gw.columns])
    gw = gw.rename(columns={
        "element": "pid", "goals_scored": "goals", "expected_goals": "xg", "expected_assists": "xa",
        "expected_goals_conceded": "xgc", "yellow_cards": "yellow", "red_cards": "red",
        "clearances_blocks_interceptions": "cbi", "defensive_contribution": "def_contrib",
        "total_points": "fpl_pts", "penalties_saved": "pen_saved", "penalties_missed": "pen_missed",
    })
    gw["season"] = season
    gw = gw.drop_duplicates(["pid", "fixture"])
    as_of = str(gw["kickoff"].max()) if len(gw) else None
    st = status(src, True, len(gw), as_of, note,
                FPL_API if src.startswith("FPL official") else f"{MIRROR_FPL}/{season}")
    return players, teams, fixtures, gw, st


# ----------------------------------------------------------------------------
# 2. football-data.co.uk (team match stats + odds)
# ----------------------------------------------------------------------------
FD_COLS = {
    "HS": "h_shots", "AS": "a_shots", "HST": "h_sot", "AST": "a_sot", "HF": "h_fouls", "AF": "a_fouls",
    "HC": "h_corners", "AC": "a_corners", "HY": "h_yellow", "AY": "a_yellow", "HR": "h_red", "AR": "a_red",
    "FTHG": "h_goals", "FTAG": "a_goals", "HTHG": "h_ht_goals", "HTAG": "a_ht_goals", "Referee": "referee",
}
# preferred odds columns, sharpest first (Pinnacle), then market average, then Bet365
ODDS_SETS = [
    ("PSCH", "PSCD", "PSCA", "PC>2.5", "PC<2.5", "Pinnacle closing"),
    ("PSH", "PSD", "PSA", "P>2.5", "P<2.5", "Pinnacle"),
    ("AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5", "Market average"),
    ("B365H", "B365D", "B365A", "B365>2.5", "B365<2.5", "Bet365"),
]


def _parse_fd_date(s):
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(str(s), fmt).date()
        except ValueError:
            continue
    return None


def _fd_frame(df, tmap):
    out = pd.DataFrame({"date": df["Date"].map(_parse_fd_date)})
    out["home"] = df["HomeTeam"].map(tmap)
    out["away"] = df["AwayTeam"].map(tmap)
    out["home_name"], out["away_name"] = df["HomeTeam"], df["AwayTeam"]
    for k, v in FD_COLS.items():
        if k in df.columns:
            out[v] = df[k]
    for h, d, a, o, u, label in ODDS_SETS:
        if h in df.columns and df[h].notna().any():
            out["odds_h"], out["odds_d"], out["odds_a"] = df[h], df[d], df[a]
            out["odds_src"] = label
            if o in df.columns:
                out["odds_o25"], out["odds_u25"] = df[o], df.get(u)
            break
    return out


def load_football_data(season, fpl_teams, current=False):
    tmap = team_mapper(fpl_teams)
    code = season_code(season)
    try:
        df = get_csv(f"{FD_BASE}/mmz4281/{code}/E0.csv")
        src, url = "football-data.co.uk", f"{FD_BASE}/mmz4281/{code}/E0.csv"
    except Exception as e:
        if current:
            return pd.DataFrame(), status("football-data.co.uk", False, note=f"unreachable ({type(e).__name__}); "
                                          "team shots/fouls/odds for this season fill in when the pipeline runs in GitHub Actions")
        try:
            df = get_csv(f"{MIRROR_FD}/season-{code}.csv")
            src, url = "football-data.co.uk via datasets GitHub mirror (no odds)", f"{MIRROR_FD}/season-{code}.csv"
        except Exception as e2:
            return pd.DataFrame(), status("football-data.co.uk", False, note=f"unreachable ({type(e2).__name__})")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    out = _fd_frame(df, tmap)
    out["season"] = season
    unmapped = sorted(set(out.loc[out.home.isna(), "home_name"]) | set(out.loc[out.away.isna(), "away_name"]))
    note = f"unmapped team names: {unmapped}" if unmapped else ""
    return out, status(src, True, len(out), str(out["date"].max()) if len(out) else None, note, url)


def load_upcoming_odds(fpl_teams):
    """football-data.co.uk/fixtures.csv lists this week's matches with bookmaker odds."""
    try:
        df = get_csv(f"{FD_BASE}/fixtures.csv")
    except Exception as e:
        return pd.DataFrame(), status("football-data.co.uk upcoming odds", False,
                                      note=f"unreachable ({type(e).__name__}); fills in from GitHub Actions")
    df = df[df["Div"] == "E0"]
    out = _fd_frame(df, team_mapper(fpl_teams))
    return out, status("football-data.co.uk upcoming odds", True, len(out), dt.date.today().isoformat(),
                       url=f"{FD_BASE}/fixtures.csv")


# ----------------------------------------------------------------------------
# player-name matching inside one fixture (small candidate sets -> reliable)
# ----------------------------------------------------------------------------
def build_name_index(players):
    idx = {}
    for _, p in players.iterrows():
        names = {norm_name(f"{p.get('first_name', '')} {p.get('second_name', '')}"),
                 norm_name(p.get("web_name", "")), norm_name(p.get("second_name", ""))}
        if isinstance(p.get("known_name"), str) and p.get("known_name"):
            names.add(norm_name(p["known_name"]))
        idx[int(p["id"])] = {n for n in names if n}
    return idx


def name_score(ext, cands):
    e = norm_name(ext)
    best = 0.0
    et = set(e.split())
    for c in cands:
        r = difflib.SequenceMatcher(None, e, c).ratio()
        ct = set(c.split())
        if ct and ct <= et or et and et <= ct:
            r = max(r, 0.9)
        if e.split()[-1:] == c.split()[-1:] and e:
            r = max(r, 0.8)
        best = max(best, r)
    return best


def match_players(ext_rows, fpl_pids, name_idx, min_score=0.62):
    """Greedy one-to-one assignment of external player names to FPL ids within a team-fixture."""
    pairs = []
    for i, (name, _mins) in enumerate(ext_rows):
        for pid in fpl_pids:
            pairs.append((name_score(name, name_idx.get(pid, set())), i, pid))
    pairs.sort(reverse=True)
    used_i, used_p, out = set(), set(), {}
    for s, i, pid in pairs:
        if s < min_score or i in used_i or pid in used_p:
            continue
        used_i.add(i)
        used_p.add(pid)
        out[i] = pid
    return out


def _fixture_lookup(fixtures):
    """(date, home_id, away_id) -> fixture id, tolerant of +-1 day timezone differences."""
    look = {}
    for _, f in fixtures.dropna(subset=["kickoff_time"]).iterrows():
        d = f["kickoff_time"].date()
        for off in (-1, 0, 1):
            look.setdefault((d + dt.timedelta(days=off), int(f["team_h"]), int(f["team_a"])), int(f["id"]))
    return look


# ----------------------------------------------------------------------------
# 3. Understat: shots, shots on target, key passes per player-match
# ----------------------------------------------------------------------------
def _understat_legacy(path, var):
    html = get(f"{UNDERSTAT}/{path}").text
    m = re.search(var + r"\s*=\s*JSON\.parse\('(.*?)'\)", html)
    if not m:
        raise ValueError(f"{var} not found")
    return json.loads(m.group(1).encode("utf-8").decode("unicode_escape"))


def _understat_api(path):
    return get(f"{UNDERSTAT}/{path}", headers={"X-Requested-With": "XMLHttpRequest"}).json()


def understat_matches(year):
    try:
        d = _understat_api(f"getLeagueData/EPL/{year}")
        return d.get("dates") or d.get("datesData")
    except Exception:
        return _understat_legacy(f"league/EPL/{year}", "datesData")


def understat_match(mid):
    try:
        d = _understat_api(f"getMatchData/{mid}")
        return d["shots"], d["rosters"]
    except Exception:
        return _understat_legacy(f"match/{mid}", "shotsData"), _understat_legacy(f"match/{mid}", "rostersData")


def load_understat(year, fixtures, players, gw, fpl_teams, season):
    tmap = team_mapper(fpl_teams)
    try:
        dates = understat_matches(year)
    except Exception as e:
        return pd.DataFrame(), status("Understat", False, note=f"unreachable ({type(e).__name__}); "
                                      "shots / shots on target / key passes fill in from GitHub Actions")
    played = [m for m in dates if str(m.get("isResult")).lower() == "true"]
    look = _fixture_lookup(fixtures)
    name_idx = build_name_index(players)
    apps = gw[gw["minutes"] > 0].groupby(["fixture", "team"])["pid"].apply(list).to_dict()

    def one(m):
        try:
            return m, understat_match(m["id"])
        except Exception:
            return m, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(one, played))

    rows, unmatched, missing_fx = [], 0, 0
    for m, payload in results:
        if not payload:
            continue
        shots, rosters = payload
        d = pd.to_datetime(m["datetime"]).date()
        h, a = tmap(m["h"]["title"]), tmap(m["a"]["title"])
        fx = look.get((d, h, a))
        if fx is None:
            missing_fx += 1
            continue
        # shots on target per understat player id (Goal + SavedShot = on target)
        sot = {}
        for side in ("h", "a"):
            for s in shots.get(side, []):
                if s.get("result") in ("Goal", "SavedShot"):
                    sot[str(s["player_id"])] = sot.get(str(s["player_id"]), 0) + 1
        for side, team in (("h", h), ("a", a)):
            roster = list(rosters.get(side, {}).values())
            ext = [(r["player"], int(r.get("time", 0))) for r in roster]
            mapping = match_players(ext, apps.get((fx, team), []), name_idx)
            for i, r in enumerate(roster):
                if i not in mapping:
                    unmatched += 1
                    continue
                rows.append({"pid": mapping[i], "fixture": fx, "shots": int(r.get("shots", 0)),
                             "sot": sot.get(str(r["player_id"]), 0), "key_passes": int(r.get("key_passes", 0)),
                             "us_xg": float(r.get("xG", 0)), "xg_chain": float(r.get("xGChain", 0)),
                             "xg_buildup": float(r.get("xGBuildup", 0))})
    df = pd.DataFrame(rows)
    note = f"{len(played)} matches; {unmatched} player rows unmatched; {missing_fx} matches not mapped to FPL fixtures"
    return df, status("Understat", len(df) > 0, len(df), max((m["datetime"] for m in played), default=None),
                      note, f"{UNDERSTAT}/league/EPL/{year}")


# ----------------------------------------------------------------------------
# 4. ESPN: fouls committed / drawn, offsides, shots, SOT per player-match
# ----------------------------------------------------------------------------
ESPN_STAT_KEYS = {
    "foulsCommitted": "fouls", "foulsSuffered": "fouls_drawn", "offsides": "offsides",
    "totalShots": "espn_shots", "shotsOnTarget": "espn_sot", "saves": "espn_saves",
}


def load_espn(season_start, fixtures, players, gw, fpl_teams):
    tmap = team_mapper(fpl_teams)
    today = dt.date.today()
    try:
        url = f"{ESPN}/scoreboard?dates={season_start:%Y%m%d}-{today:%Y%m%d}&limit=500"
        events = get(url).json().get("events", [])
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), status("ESPN", False, note=f"unreachable ({type(e).__name__}); "
                                                     "fouls / fouls drawn fill in from GitHub Actions")
    done = [e for e in events if e.get("status", {}).get("type", {}).get("completed")]
    look = _fixture_lookup(fixtures)
    name_idx = build_name_index(players)
    apps = gw[gw["minutes"] > 0].groupby(["fixture", "team"])["pid"].apply(list).to_dict()

    def one(e):
        try:
            return e, get(f"{ESPN}/summary?event={e['id']}").json()
        except Exception:
            return e, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(one, done))

    prow, trow, unmatched = [], [], 0
    for e, summ in results:
        if not summ:
            continue
        comp = e["competitions"][0]
        side = {c["homeAway"]: c for c in comp["competitors"]}
        h, a = tmap(side["home"]["team"]["displayName"]), tmap(side["away"]["team"]["displayName"])
        d = pd.to_datetime(e["date"]).date()
        fx = look.get((d, h, a))
        if fx is None:
            continue
        for t in summ.get("boxscore", {}).get("teams", []):
            team = tmap(t["team"]["displayName"])
            stats = {s.get("name"): s.get("displayValue") for s in t.get("statistics", [])}
            trow.append({"fixture": fx, "team": team, "possession": pd.to_numeric(stats.get("possessionPct"), errors="coerce")})
        for ros in summ.get("rosters", []):
            team = tmap(ros["team"]["displayName"])
            plist = [p for p in ros.get("roster", []) if p.get("stats")]
            ext = [(p["athlete"]["displayName"], 0) for p in plist]
            mapping = match_players(ext, apps.get((fx, team), []), name_idx)
            for i, p in enumerate(plist):
                st = {s.get("name"): s.get("value") for s in p.get("stats", [])}
                if not st.get("appearances"):
                    continue
                if i not in mapping:
                    unmatched += 1
                    continue
                row = {"pid": mapping[i], "fixture": fx}
                for k, v in ESPN_STAT_KEYS.items():
                    row[v] = pd.to_numeric(st.get(k), errors="coerce")
                prow.append(row)
    pdf = pd.DataFrame(prow)
    return pdf, pd.DataFrame(trow), status("ESPN", len(pdf) > 0, len(pdf), str(today),
                                           f"{len(done)} completed matches; {unmatched} player rows unmatched", ESPN)
