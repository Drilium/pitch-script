"""
Projection model for Pitch Script.

Everything here works from a cutoff time, using only matches that kicked off
before it. The same code builds this week's projections and the walk-forward
backtest, so the accuracy numbers on the site describe the model that is live.

Pieces
------
1. Team strength   - attack / defence ratings from xG (blended with goals),
                     opponent-adjusted, shrunk toward last season.
2. Match goals     - Poisson score grid -> win/draw/loss, over/under, BTTS, clean sheets.
3. Player minutes  - chance to start, minutes when starting, chance of a sub
                     appearance, availability from FPL injury status.
4. Player rates    - per-90 rates for every prop stat, shrunk toward the player's
                     own last season and then the position average.
5. Matchup factor  - attacking stats scale with the team's expected goals in
                     this matchup; defensive / discipline / keeper stats scale with
                     what this opponent gives up to that position.
6. Probabilities   - negative binomial (Poisson with extra spread, fitted per
                     stat from real data) for every over/under line.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# stat key -> settings
#   cols    : how the per-match value is built from raw columns
#   m       : prior strength in 90-minute units (higher = trust sample less)
#   adj     : 'att' (scales with team xG in this matchup) or 'opp' (opponent allows to position)
#   lines   : sportsbook lines shown
STATS = {
    "goals":   {"label": "Goals",            "m": 6,  "adj": "att", "lines": [0.5, 1.5]},
    "assists": {"label": "Assists",          "m": 6,  "adj": "att", "lines": [0.5]},
    "ga":      {"label": "Goal or assist",   "m": 6,  "adj": "att", "lines": [0.5, 1.5]},
    "shots":   {"label": "Shots",            "m": 5,  "adj": "att", "lines": [0.5, 1.5, 2.5, 3.5]},
    "sot":     {"label": "Shots on target",  "m": 5,  "adj": "att", "lines": [0.5, 1.5, 2.5]},
    "tackles": {"label": "Tackles",          "m": 5,  "adj": "opp", "lines": [0.5, 1.5, 2.5, 3.5]},
    "fouls":   {"label": "Fouls committed",  "m": 6,  "adj": "opp", "lines": [0.5, 1.5, 2.5]},
    "cards":   {"label": "Carded",           "m": 15, "adj": "opp", "lines": [0.5]},
    "saves":   {"label": "Keeper saves",     "m": 4,  "adj": "opp", "lines": [0.5, 1.5, 2.5, 3.5, 4.5]},
    "def_contrib": {"label": "Defensive contributions", "m": 5, "adj": "opp", "lines": [9.5, 11.5]},
}
POSITIONS = ["GK", "DEF", "MID", "FWD"]
PREV_WEIGHT = 0.6          # last season's minutes count 60% as much as this season's
TEAM_K = 12.0              # team ratings: last season worth ~12 matches of evidence
PRIOR_REGRESS = 0.4        # carry 40% of last season's distance from average into this season
TOTAL_W = 0.3              # match totals: 30% team-specific, 70% league average (backtest: totals barely predictable)
RHO = -0.10                # Dixon-Coles low-score correction (fixes Poisson's draw underestimate)
ATT_W = 1.0                # how much of the opponent-defence effect passes to player attacking stats
OPP_C = 20.0               # opponent factor shrink (90s)
OPP_C_STAT = {"cards": 80.0}   # cards are mostly referee/game-state noise -> trust opponent effect less
DECAY_HALF_LIFE = 8.0      # matches; recent selection matters more for minutes


# ----------------------------------------------------------------------------
# derived per-match stat columns
# ----------------------------------------------------------------------------
def add_stat_columns(pm: pd.DataFrame) -> pd.DataFrame:
    pm = pm.copy()
    g = lambda c: pm[c] if c in pm.columns else pd.Series(np.nan, index=pm.index)
    # goals/assists use a blend: mostly expected (more stable), some actual (finishing / creation skill)
    pm["goals_v"] = 0.85 * g("xg") + 0.15 * g("goals")
    pm["assists_v"] = 0.8 * g("xa") + 0.2 * g("assists")
    pm["ga_v"] = pm["goals_v"] + pm["assists_v"]
    # shots prefer Understat, fall back to ESPN
    pm["shots_v"] = g("shots").fillna(g("espn_shots"))
    pm["sot_v"] = g("sot").fillna(g("espn_sot"))
    pm["tackles_v"] = g("tackles")
    pm["fouls_v"] = g("fouls")
    pm["cards_v"] = g("yellow").fillna(0) + g("red").fillna(0)
    pm["saves_v"] = g("saves")
    pm["def_contrib_v"] = g("def_contrib")
    # the raw outcome we grade against (goals/assists are counts, not the blend)
    pm["goals_o"], pm["assists_o"] = g("goals"), g("assists")
    pm["ga_o"] = g("goals") + g("assists")
    for s in ["shots", "sot", "tackles", "fouls", "cards", "saves", "def_contrib"]:
        pm[f"{s}_o"] = pm[f"{s}_v"]
    return pm


# ----------------------------------------------------------------------------
# distributions
# ----------------------------------------------------------------------------
def nb_sf(k: float, mu: float, r: float | None) -> float:
    """P(X >= k) for X ~ NegBin(mean mu, size r); r=None -> Poisson."""
    k = int(math.ceil(k))
    if mu <= 0:
        return 0.0 if k > 0 else 1.0
    if k <= 0:
        return 1.0
    if not r or r > 1e4:
        p, term = 0.0, math.exp(-mu)
        for i in range(k):
            p += term
            term *= mu / (i + 1)
        return max(0.0, min(1.0, 1 - p))
    q = r / (r + mu)
    term = q ** r
    cdf = 0.0
    for i in range(k):
        cdf += term
        term *= (i + r) / (i + 1) * (1 - q)
    return max(0.0, min(1.0, 1 - cdf))


def fit_dispersion(pm: pd.DataFrame) -> dict:
    """Pool regular starters' match-to-match spread to fit NB size per stat (method of moments)."""
    out = {}
    starters = pm[(pm["minutes"] >= 75)]
    for s in STATS:
        col = f"{s}_o"
        if col not in starters or starters[col].notna().sum() < 500:
            out[s] = None
            continue
        d = starters.dropna(subset=[col])
        grp = d.groupby("code")[col]
        stats = pd.DataFrame({"n": grp.size(), "mean": grp.mean(), "var": grp.var()})
        stats = stats[stats["n"] >= 10]
        if stats.empty:
            out[s] = None
            continue
        mu = (stats["mean"] * stats["n"]).sum() / stats["n"].sum()
        mu2 = (stats["mean"] ** 2 * stats["n"]).sum() / stats["n"].sum()
        var = (stats["var"] * stats["n"]).sum() / stats["n"].sum()
        extra = var - (stats["mean"] * stats["n"]).sum() / stats["n"].sum()
        out[s] = round(float(mu2 / extra), 2) if extra > 1e-6 else None
    return out


# ----------------------------------------------------------------------------
# team strength
# ----------------------------------------------------------------------------
def team_match_table(pm: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per team per finished fixture with xG for/against and goals for/against."""
    xg = pm.groupby(["season", "fixture", "team"], as_index=False)["xg"].sum()
    f = fixtures[fixtures["finished"].astype(str).str.lower().isin(["true", "1"])][
        ["season", "id", "kickoff_time", "team_h", "team_a", "team_h_score", "team_a_score"]].dropna()
    home = pd.DataFrame({"season": f.season, "fixture": f.id, "kickoff": f.kickoff_time, "team": f.team_h,
                         "opp": f.team_a, "home": True, "gf": f.team_h_score, "ga": f.team_a_score})
    away = pd.DataFrame({"season": f.season, "fixture": f.id, "kickoff": f.kickoff_time, "team": f.team_a,
                         "opp": f.team_h, "home": False, "gf": f.team_a_score, "ga": f.team_h_score})
    t = pd.concat([home, away], ignore_index=True)
    for c in ["fixture", "team", "opp"]:
        t[c] = t[c].astype(int)
    xg = xg.astype({"fixture": int, "team": int})
    t = t.merge(xg.rename(columns={"xg": "xgf"}), on=["season", "fixture", "team"], how="left")
    t = t.merge(xg.rename(columns={"team": "opp", "xg": "xga"}), on=["season", "fixture", "opp"], how="left")
    t["sf"] = 0.75 * t["xgf"] + 0.25 * t["gf"]   # scoring signal
    t["sa"] = 0.75 * t["xga"] + 0.25 * t["ga"]
    return t.dropna(subset=["sf", "sa"])


def fit_team_ratings(tm: pd.DataFrame, teams: list[int], prior: dict | None = None, k: float = TEAM_K, iters: int = 15):
    """Multiplicative attack/defence ratings, opponent-adjusted, shrunk toward prior (default 1.0)."""
    if tm.empty:
        return {"att": {t: (prior or {}).get("att", {}).get(t, 1.0) for t in teams},
                "def": {t: (prior or {}).get("def", {}).get(t, 1.0) for t in teams},
                "mu_home": 1.5, "mu_away": 1.2}
    mu_home = tm.loc[tm.home, "sf"].mean()
    mu_away = tm.loc[~tm.home, "sf"].mean()
    mu = (mu_home + mu_away) / 2
    pa = (prior or {}).get("att", {})
    pd_ = (prior or {}).get("def", {})
    att = {t: pa.get(t, 1.0) for t in teams}
    de = {t: pd_.get(t, 1.0) for t in teams}
    tm = tm.copy()
    tm["mu_side"] = np.where(tm.home, mu_home, mu_away)
    tm["mu_opp_side"] = np.where(tm.home, mu_away, mu_home)
    for _ in range(iters):
        tm["d_opp"] = tm["opp"].map(de).fillna(1.0)
        tm["a_opp"] = tm["opp"].map(att).fillna(1.0)
        g = tm.groupby("team")
        num_a, den_a = g["sf"].sum(), (tm["mu_side"] * tm["d_opp"]).groupby(tm["team"]).sum()
        num_d, den_d = g["sa"].sum(), (tm["mu_opp_side"] * tm["a_opp"]).groupby(tm["team"]).sum()
        for t in teams:
            att[t] = (num_a.get(t, 0) + k * mu * pa.get(t, 1.0)) / (den_a.get(t, 0) + k * mu)
            de[t] = (num_d.get(t, 0) + k * mu * pd_.get(t, 1.0)) / (den_d.get(t, 0) + k * mu)
        # normalise so the league average is 1.0
        ma, md = np.mean(list(att.values())), np.mean(list(de.values()))
        att = {t: v / ma for t, v in att.items()}
        de = {t: v / md for t, v in de.items()}
    return {"att": att, "def": de, "mu_home": float(mu_home), "mu_away": float(mu_away),
            "league_tot": float(tm["gf"].sum() / (len(tm) / 2))}


def prior_from_last_season(prev_ratings: dict, cur_teams: list[int], code_map: dict):
    """Carry last season's ratings to this season's team ids; promoted sides get the
    average of the three lowest-rated returning... -> of last season's relegated-style teams."""
    if not prev_ratings:
        return None
    att_prev, def_prev = prev_ratings["att"], prev_ratings["def"]
    # code_map: current team id -> previous season team id (None for promoted clubs)
    net = {t: att_prev[t] / def_prev[t] for t in att_prev}
    bottom3 = sorted(net, key=net.get)[:3]
    promo_att = float(np.mean([att_prev[t] for t in bottom3]))
    promo_def = float(np.mean([def_prev[t] for t in bottom3]))
    att, de = {}, {}
    for t in cur_teams:
        pt = code_map.get(t)
        if pt is not None and pt in att_prev:
            # regress last season 1/3 toward average (squads change in the summer)
            att[t] = 1 + (att_prev[pt] - 1) * PRIOR_REGRESS
            de[t] = 1 + (def_prev[pt] - 1) * PRIOR_REGRESS
        else:
            att[t], de[t] = promo_att, promo_def
    return {"att": att, "def": de}


def score_grid(lh, la, maxg=10):
    ph = np.array([math.exp(-lh) * lh ** i / math.factorial(i) for i in range(maxg + 1)])
    pa = np.array([math.exp(-la) * la ** i / math.factorial(i) for i in range(maxg + 1)])
    return np.outer(ph, pa)


def match_markets(lh, la, rho=RHO):
    g = score_grid(lh, la)
    tau = np.ones_like(g)
    tau[0, 0], tau[0, 1], tau[1, 0], tau[1, 1] = 1 - lh * la * rho, 1 + lh * rho, 1 + la * rho, 1 - rho
    g = g * tau
    g /= g.sum()
    idx = np.add.outer(np.arange(g.shape[0]), np.arange(g.shape[1]))
    return {
        "lam_h": round(lh, 3), "lam_a": round(la, 3),
        "p_home": float(np.tril(g, -1).sum()), "p_draw": float(np.trace(g)), "p_away": float(np.triu(g, 1).sum()),
        "o15": float(g[idx >= 2].sum()), "o25": float(g[idx >= 3].sum()), "o35": float(g[idx >= 4].sum()),
        "btts": float(1 - g[0, :].sum() - g[:, 0].sum() + g[0, 0]),
        "cs_h": float(g[:, 0].sum()), "cs_a": float(g[0, :].sum()),
        "top_scores": sorted(([f"{i}-{j}", float(g[i, j])] for i in range(6) for j in range(6)),
                             key=lambda x: -x[1])[:5],
    }


# ----------------------------------------------------------------------------
# player minutes
# ----------------------------------------------------------------------------
def availability(status, chance):
    if status in ("i", "s", "u", "n"):
        return 0.0
    if chance is not None and not (isinstance(chance, float) and math.isnan(chance)):
        return float(chance) / 100.0
    return 1.0 if status in ("a", None) else 0.5


def minutes_model(pm_cur: pd.DataFrame, team_sched: pd.DataFrame, cur_team: dict, prev_share: dict):
    """
    pm_cur     : this season's rows before cutoff
    team_sched : finished fixtures before cutoff (season, fixture, team, kickoff)
    cur_team   : code -> current team id
    prev_share : code -> last season start share (prior)
    Returns code -> dict(p_start, ms, p_sub, sm)
    """
    out = {}
    rows = pm_cur.set_index(["code", "fixture"])[["minutes", "starts"]]
    sched = team_sched.sort_values("kickoff")
    by_team = {t: g["fixture"].tolist() for t, g in sched.groupby("team")}
    look = rows.to_dict("index")
    for code, team in cur_team.items():
        fx = by_team.get(team, [])
        n = len(fx)
        w = np.array([0.5 ** ((n - 1 - i) / DECAY_HALF_LIFE) for i in range(n)]) if n else np.array([])
        st, sb, ms, sm = [], [], [], []
        for f in fx:
            r = look.get((code, f))
            mins = r["minutes"] if r else 0
            start = bool(r and r["starts"] >= 1)
            st.append(1.0 if start else 0.0)
            sb.append(1.0 if (not start and mins > 0) else 0.0)
            if start:
                ms.append(mins)
            elif mins > 0:
                sm.append(mins)
        prior = prev_share.get(code, 0.3)
        k0 = 3.0
        if n:
            ws = float(w.sum())
            p_start = (float((w * np.array(st)).sum()) + k0 * prior) / (ws + k0)
            not_start_w = float((w * (1 - np.array(st))).sum())
            p_sub = (float((w * np.array(sb)).sum()) + 1.0 * 0.3) / (not_start_w + 1.0)
        else:
            p_start, p_sub = prior, 0.3
        out[code] = {
            "p_start": round(p_start, 3), "p_sub": round(p_sub, 3),
            "ms": round(float(np.mean(ms)) if len(ms) >= 2 else (np.mean(ms) * 0.5 + 80 * 0.5 if ms else 80.0), 1),
            "sm": round(float(np.mean(sm)) if sm else 20.0, 1),
            "apps": int(sum(st) + sum(sb)), "starts": int(sum(st)), "team_matches": n,
        }
    return out


# ----------------------------------------------------------------------------
# player rates
# ----------------------------------------------------------------------------
def player_rates(pm_cur, pm_prev, pos_of: dict):
    """code -> stat -> dict(rate90, cur_n90, cur_total, prev_rate90)"""
    pooled = pd.concat([pm_cur, pm_prev], ignore_index=True)
    pooled = pooled[pooled["minutes"] > 0].copy()
    pooled["pos"] = pooled["code"].map(pos_of)
    rates = {}
    pos_avg = {}
    for s in STATS:
        col = f"{s}_v"
        d = pooled.dropna(subset=[col])
        g = d.groupby("pos").agg(v=(col, "sum"), m=("minutes", "sum"))
        pos_avg[s] = {p: float(g.loc[p, "v"] / (g.loc[p, "m"] / 90)) if p in g.index and g.loc[p, "m"] > 0 else 0.0
                      for p in POSITIONS}

    def agg(df, col):
        d = df[df["minutes"] > 0].dropna(subset=[col])
        g = d.groupby("code").agg(v=(col, "sum"), m=("minutes", "sum"))
        return {c: (float(r.v), float(r.m)) for c, r in g.iterrows()}

    for s, cfg in STATS.items():
        col = f"{s}_v"
        cur, prev = agg(pm_cur, col), agg(pm_prev, col)
        m = cfg["m"]
        for code, pos in pos_of.items():
            base = pos_avg[s].get(pos, 0.0)
            pv, pmin = prev.get(code, (0.0, 0.0))
            if pmin > 0:
                pm90 = pmin / 90
                prior = (PREV_WEIGHT * pv + m * base) / (PREV_WEIGHT * pm90 + m)
                prev_rate = pv / pm90
            else:
                prior, prev_rate, pm90 = base, None, 0.0
            cv, cmin = cur.get(code, (0.0, 0.0))
            c90 = cmin / 90
            rate = (cv + m * prior) / (c90 + m)
            rates.setdefault(code, {})[s] = {
                "rate90": rate, "cur_n90": c90, "cur_total": float(cv),
                "cur_rate90": (cv / c90) if c90 > 0 else None, "prev_rate90": prev_rate, "prev_n90": pm90,
            }
    return rates, pos_avg


def opponent_factors(pm_cur, pm_prev, pos_of, pos_avg, teams_cur, prev_to_cur_team):
    """team -> pos -> stat -> factor (what this opponent gives up vs league average)."""
    p = pm_prev.copy()
    p["opp"] = p["opp"].map(prev_to_cur_team)
    p["w"] = 0.5
    c = pm_cur.copy()
    c["w"] = 1.0
    d = pd.concat([c, p.dropna(subset=["opp"])], ignore_index=True)
    d = d[d["minutes"] > 0].copy()
    d["pos"] = d["code"].map(pos_of)
    d["opp"] = d["opp"].astype(int)
    out = {t: {pos: {} for pos in POSITIONS} for t in teams_cur}
    for s, cfg in STATS.items():
        if cfg["adj"] != "opp":
            continue
        col = f"{s}_v"
        dd = d.dropna(subset=[col]).copy()
        dd["wv"] = dd[col] * dd["w"]
        dd["wm"] = dd["minutes"] * dd["w"] / 90
        g = dd.groupby(["opp", "pos"]).agg(v=("wv", "sum"), n=("wm", "sum"))
        for t in teams_cur:
            for pos in POSITIONS:
                base = pos_avg[s].get(pos, 0)
                if base <= 0:
                    out[t][pos][s] = 1.0
                    continue
                if (t, pos) in g.index:
                    v, n = g.loc[(t, pos), "v"], g.loc[(t, pos), "n"]
                else:
                    v, n = 0.0, 0.0
                c = OPP_C_STAT.get(s, OPP_C)
                out[t][pos][s] = float((v + c * base) / (base * (n + c)))
    return out


def home_factors(pm_all):
    d = pm_all[pm_all["minutes"] > 0]
    out = {}
    for s in STATS:
        col = f"{s}_v"
        dd = d.dropna(subset=[col])
        if dd.empty:
            out[s] = (1.0, 1.0)
            continue
        r = lambda x: x[col].sum() / (x["minutes"].sum() / 90)
        overall = r(dd)
        h, a = r(dd[dd.was_home]), r(dd[~dd.was_home])
        out[s] = (h / overall if overall else 1.0, a / overall if overall else 1.0)
    return out


# ----------------------------------------------------------------------------
# full projection at a cutoff
# ----------------------------------------------------------------------------
def build_context(pm_cur, pm_prev, fx_cur, fx_prev, players_cur, cutoff, prev_team_ratings=None,
                  team_code_cur=None, team_code_prev=None):
    """Everything the projections need, using only data before `cutoff`."""
    pm_c = pm_cur[pm_cur["kickoff"] < cutoff]
    fx_c = fx_cur.copy()
    fx_c.loc[fx_c["kickoff_time"] >= cutoff, "finished"] = False
    teams_cur = sorted(players_cur["team"].astype(int).unique().tolist())
    # team ids across seasons are linked through FPL team 'code'
    prev_by_code = {v: k for k, v in (team_code_prev or {}).items()}
    cur_to_prev = {t: prev_by_code.get(c) for t, c in (team_code_cur or {}).items()}
    prev_to_cur = {p: t for t, p in cur_to_prev.items() if p is not None}

    tm_c = team_match_table(pm_c, fx_c)
    prior = prior_from_last_season(prev_team_ratings, teams_cur, cur_to_prev) if prev_team_ratings else None
    ratings = fit_team_ratings(tm_c, teams_cur, prior)
    # league goals per match: this season blended with ~100 matches of last season
    n_cur = len(tm_c) / 2
    prev_tot = (prev_team_ratings or {}).get("league_tot", 2.8)
    cur_tot = tm_c["gf"].sum() / n_cur if n_cur else prev_tot
    ratings["league_tot"] = float((cur_tot * n_cur + prev_tot * 100) / (n_cur + 100))
    # use pooled home/away means (this season is small early on)
    if prev_team_ratings and len(tm_c) < 200:
        w = len(tm_c) / 200
        ratings["mu_home"] = w * ratings["mu_home"] + (1 - w) * prev_team_ratings["mu_home"]
        ratings["mu_away"] = w * ratings["mu_away"] + (1 - w) * prev_team_ratings["mu_away"]

    pos_of = dict(zip(players_cur["code"], players_cur["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})))
    cur_team = dict(zip(players_cur["code"], players_cur["team"].astype(int)))
    rates, pos_avg = player_rates(pm_c, pm_prev, pos_of)
    oppf = opponent_factors(pm_c, pm_prev, pos_of, pos_avg, teams_cur, prev_to_cur)
    homef = home_factors(pd.concat([pm_c, pm_prev]))

    # last season start share as prior for selection
    prev_share = {}
    if len(pm_prev):
        tp = pm_prev.groupby("code").agg(st=("starts", "sum"), team=("team", "last"))
        team_games = pm_prev.groupby("team")["fixture"].nunique()
        prev_share = {c: min(0.95, r.st / team_games.get(r.team, 38)) for c, r in tp.iterrows()}
    sched = tm_c[tm_c["season"] == pm_cur["season"].iloc[0]][["fixture", "team", "kickoff"]] if len(tm_c) else \
        pd.DataFrame(columns=["fixture", "team", "kickoff"])
    mins = minutes_model(pm_c, sched, cur_team, prev_share)
    return {"ratings": ratings, "rates": rates, "pos_avg": pos_avg, "oppf": oppf, "homef": homef,
            "mins": mins, "pos_of": pos_of, "cur_team": cur_team}


def project_fixture(ctx, fixture_row, players_cur, dispersion, use_availability=True):
    """Team markets + every player's prop probabilities for one fixture."""
    R = ctx["ratings"]
    h, a = int(fixture_row["team_h"]), int(fixture_row["team_a"])
    lh = R["mu_home"] * R["att"][h] * R["def"][a]
    la = R["mu_away"] * R["att"][a] * R["def"][h]
    mu_bar = (R["mu_home"] + R["mu_away"]) / 2
    tot = TOTAL_W * (lh + la) + (1 - TOTAL_W) * R.get("league_tot", lh + la)
    lh, la = lh * tot / (lh + la), la * tot / (lh + la)
    team_out = match_markets(lh, la)
    rows = []
    sub = players_cur[players_cur["team"].isin([h, a])]
    for _, p in sub.iterrows():
        code, team = p["code"], int(p["team"])
        home = team == h
        opp = a if home else h
        pos = ctx["pos_of"].get(code)
        mm = ctx["mins"].get(code)
        if mm is None or pos is None:
            continue
        avail = availability(p.get("status"), p.get("chance_of_playing_next_round")) if use_availability else 1.0
        if avail <= 0 and use_availability:
            continue
        f_att = (R["mu_home"] if home else R["mu_away"]) / mu_bar * (1 + ATT_W * (R["def"][opp] - 1))
        markets = {}
        for s, cfg in STATS.items():
            if s == "saves" and pos != "GK":
                continue
            if pos == "GK" and s in ("def_contrib", "tackles", "fouls", "shots", "sot", "goals", "assists", "ga"):
                continue
            r = ctx["rates"][code][s]
            if r["rate90"] is None or (r["cur_n90"] == 0 and r["prev_rate90"] is None and s in ("shots", "sot", "fouls")):
                # no real data for this stat for this player -> don't invent one
                if ctx["pos_avg"][s].get(pos, 0) == 0:
                    continue
            hf = ctx["homef"][s][0 if home else 1]
            if cfg["adj"] == "att":
                fac = f_att
            else:
                fac = ctx["oppf"].get(opp, {}).get(pos, {}).get(s, 1.0) * hf
            r90 = r["rate90"] * fac
            lam_s = r90 * mm["ms"] / 90
            lam_b = r90 * mm["sm"] / 90
            disp = dispersion.get(s)
            lines = []
            for L in cfg["lines"]:
                ps = nb_sf(L, lam_s, disp)
                pb = nb_sf(L, lam_b, disp)
                p_play = mm["p_start"] + (1 - mm["p_start"]) * mm["p_sub"]
                p_if_plays = (mm["p_start"] * ps + (1 - mm["p_start"]) * mm["p_sub"] * pb) / p_play if p_play > 0 else 0
                lines.append([L, round(ps, 4), round(p_if_plays, 4)])
            markets[s] = {"lam": round(lam_s, 3), "r90": round(r["rate90"], 3), "fac": round(fac, 3), "lines": lines}
        if not markets:
            continue
        rows.append({
            "pid": int(p["id"]), "code": int(code), "team": team, "opp": opp, "home": home, "pos": pos,
            "avail": round(avail, 2), "p_start": mm["p_start"], "p_sub": mm["p_sub"], "ms": mm["ms"], "sm": mm["sm"],
            "em": round(avail * (mm["p_start"] * mm["ms"] + (1 - mm["p_start"]) * mm["p_sub"] * mm["sm"]), 1),
            "m": markets,
        })
    return team_out, rows


# ----------------------------------------------------------------------------
# calibration layer (Platt scaling per stat, fitted on last season's backtest)
# ----------------------------------------------------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_platt(p, y, iters=50):
    """Logistic regression y ~ a + b*logit(p) via Newton's method."""
    x = _logit(p)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    w = np.array([0.0, 1.0])
    for _ in range(iters):
        z = X @ w
        mu = 1 / (1 + np.exp(-z))
        g = X.T @ (y - mu)
        H = -(X * (mu * (1 - mu))[:, None]).T @ X - 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-8:
            break
    return [round(float(w[0]), 4), round(float(w[1]), 4)]


def apply_platt(p, ab):
    if not ab:
        return p
    z = ab[0] + ab[1] * float(_logit(p))
    return float(1 / (1 + math.exp(-z)))


def calibrate_rows(rows, calib):
    """Apply per-stat calibration to projection rows in place (both 'if starts' and 'if plays')."""
    if not calib:
        return rows
    for r in rows:
        for s, mk in r["m"].items():
            ab = calib.get(s)
            if not ab:
                continue
            mk["lines"] = [[L, round(apply_platt(ps, ab), 4), round(apply_platt(pp, ab), 4)] for L, ps, pp in mk["lines"]]
    return rows
