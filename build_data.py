#!/usr/bin/env python3
"""
Pitch Script - weekly data refresh.

    python build_data.py               # refresh data + projections (uses cached backtest)
    python build_data.py --backtest    # also re-run the last-season walk-forward backtest

Writes soccer_data.json (read by index.html at runtime).
Last season's external data (Understat / ESPN) is cached in cache/ after the
first successful pull, so weekly runs only fetch the current season.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import model as M
import sources as S

CUR, PREV, PREV2 = "2026-27", "2025-26", "2024-25"
CUR_START = dt.date(2026, 8, 1)
PREV_START = dt.date(2025, 8, 1)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "soccer_data.json")
BACKTEST_OUT = os.path.join(CACHE, "backtest.json")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def add_codes(pm, players, season):
    pm = pm.copy()
    pm["code"] = pm["pid"].map(dict(zip(players["id"], players["code"])))
    pm["season"] = season
    return pm.dropna(subset=["code"]).astype({"code": int})


def cached(name, loader):
    """Load a cached CSV if present; otherwise run loader() and cache non-empty results."""
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df, S.status(f"{name} (cached)", True, len(df), note="cached after first successful pull")
    df, st = loader()
    if len(df):
        os.makedirs(CACHE, exist_ok=True)
        df.to_csv(path, index=False)
    return df, st


def merge_external(pm, ext):
    if ext is None or not len(ext):
        return pm
    ext = ext.drop_duplicates(["pid", "fixture"])
    return pm.merge(ext, on=["pid", "fixture"], how="left")


# ----------------------------------------------------------------------------
# health checks: do independent sources agree?
# ----------------------------------------------------------------------------
def health_checks(pm, fixtures, fd, season):
    out = []
    fin = fixtures[fixtures["finished"].astype(str).str.lower().isin(["true", "1"])].dropna(subset=["team_h_score"])
    # 1. player goals (+ opponent own goals) reconcile to the final score
    g = pm.groupby(["fixture", "team"]).agg(goals=("goals", "sum"), og=("own_goals", "sum")).reset_index()
    ok = bad = 0
    bad_list = []
    for _, f in fin.iterrows():
        def side(t, o):
            gt = g[(g.fixture == f.id) & (g.team == t)]["goals"].sum()
            og = g[(g.fixture == f.id) & (g.team == o)]["og"].sum()
            return gt + og
        hs, as_ = side(f.team_h, f.team_a), side(f.team_a, f.team_h)
        if hs == f.team_h_score and as_ == f.team_a_score:
            ok += 1
        else:
            bad += 1
            bad_list.append(int(f.id))
    out.append({"season": season, "check": "Player goals + own goals add up to every final score",
                "passed": ok, "failed": bad, "detail": f"fixtures failing: {bad_list[:10]}" if bad else ""})
    # 2. a team never has more than 11 starters, every match has 11 per side
    st = pm.groupby(["fixture", "team"])["starts"].sum()
    out.append({"season": season, "check": "Exactly 11 starters per team per match",
                "passed": int((st == 11).sum()), "failed": int((st != 11).sum()),
                "detail": "" if (st == 11).all() else f"examples: {st[st != 11].head(5).to_dict()}"})
    # 3. football-data scores agree with FPL scores (two independent feeds)
    if fd is not None and len(fd):
        f2 = fin.copy()
        f2["date"] = f2["kickoff_time"].dt.date
        m = f2.merge(fd, left_on=["team_h", "team_a"], right_on=["home", "away"], how="inner")
        agree = ((m.team_h_score == m.h_goals) & (m.team_a_score == m.a_goals)).sum()
        out.append({"season": season, "check": "football-data.co.uk scores match FPL scores",
                    "passed": int(agree), "failed": int(len(m) - agree),
                    "detail": f"{len(m)} of {len(fin)} finished fixtures matched across sources"})
    # 4. shots data sanity (only when Understat present): SOT <= shots, goals <= SOT
    if "shots" in pm.columns and pm["shots"].notna().any():
        d = pm.dropna(subset=["shots", "sot"])
        bad = int(((d.sot > d.shots) | (d.goals > d.sot)).sum())
        out.append({"season": season, "check": "Shots on target <= shots and goals <= shots on target",
                    "passed": int(len(d) - bad), "failed": bad, "detail": "Understat counts vs FPL goals"})
    return out


# ----------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------
def clean(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if (math.isnan(v) or math.isinf(v)) else round(float(v), 3)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        return v.isoformat()
    if v is pd.NA:
        return None
    return v


def deep_clean(o):
    """Recursively turn NaN/inf/numpy/pandas values into JSON-safe ones."""
    if isinstance(o, dict):
        return {str(k): deep_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [deep_clean(v) for v in o]
    if isinstance(o, str) or o is None or isinstance(o, bool):
        return o
    return clean(o)


LOG_COLS = ["season", "gw", "fixture", "kickoff", "code", "team_s", "opp_s", "was_home", "minutes", "starts",
            "goals", "assists", "xg", "xa", "shots", "sot", "key_passes", "tackles", "cbi", "recoveries",
            "def_contrib", "fouls", "fouls_drawn", "yellow", "red", "saves", "goals_conceded", "xgc",
            "bonus", "fpl_pts", "score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true", help="re-run last-season backtest")
    args = ap.parse_args()
    now = pd.Timestamp.now(tz="UTC")
    sources = []

    # ---------------- FPL -----------------
    log("FPL current season ...")
    pl_c, tm_c, fx_c, pm_c, st = S.load_fpl(CUR, current=True); sources.append(st)
    log("FPL previous seasons ...")
    pl_p, tm_p, fx_p, pm_p, st = S.load_fpl(PREV); sources.append(st)
    pl_p2, tm_p2, fx_p2, pm_p2, st = S.load_fpl(PREV2); sources.append(st)
    fx_c["season"], fx_p["season"], fx_p2["season"] = CUR, PREV, PREV2
    pm_c, pm_p, pm_p2 = add_codes(pm_c, pl_c, CUR), add_codes(pm_p, pl_p, PREV), add_codes(pm_p2, pl_p2, PREV2)

    # ---------------- Understat / ESPN -----------------
    log("Understat ...")
    us_c, st = S.load_understat(2026, fx_c, pl_c, pm_c, tm_c, CUR); sources.append(st)
    us_p, st = cached("understat_2025-26.csv", lambda: S.load_understat(2025, fx_p, pl_p, pm_p, tm_p, PREV)); sources.append(st)
    log("ESPN ...")
    es_c, poss_c, st = S.load_espn(CUR_START, fx_c, pl_c, pm_c, tm_c); sources.append(st)
    es_p, st = cached("espn_2025-26.csv", lambda: S.load_espn(PREV_START, fx_p, pl_p, pm_p, tm_p)[::2]); sources.append(st)
    pm_c = merge_external(merge_external(pm_c, us_c), es_c)
    pm_p = merge_external(merge_external(pm_p, us_p), es_p)

    # ---------------- football-data -----------------
    log("football-data ...")
    fd_c, st = S.load_football_data(CUR, tm_c, current=True); sources.append(st)
    fd_p, st = S.load_football_data(PREV, tm_p); sources.append(st)
    odds_up, st = S.load_upcoming_odds(tm_c); sources.append(st)

    # ---------------- checks -----------------
    health = health_checks(pm_c, fx_c, fd_c, CUR) + health_checks(pm_p, fx_p, fd_p, PREV)
    for h in health:
        log(f"  [{h['season']}] {h['check']}: {h['passed']} ok / {h['failed']} failed {h['detail']}")

    # ---------------- model -----------------
    pm_c, pm_p, pm_p2 = M.add_stat_columns(pm_c), M.add_stat_columns(pm_p), M.add_stat_columns(pm_p2)
    for d in (pm_c, pm_p, pm_p2):
        d["was_home"] = d["was_home"].astype(bool)
    tcode_c = dict(zip(tm_c["id"], tm_c["code"]))
    tcode_p = dict(zip(tm_p["id"], tm_p["code"]))
    tcode_p2 = dict(zip(tm_p2["id"], tm_p2["code"]))

    # last season's final ratings (with 2024-25 as their prior) become this season's prior
    tm_tab_p2 = M.team_match_table(pm_p2, fx_p2)
    r_p2 = M.fit_team_ratings(tm_tab_p2, sorted(tm_p2["id"].astype(int)))
    prior_p = M.prior_from_last_season(r_p2, sorted(tm_p["id"].astype(int)),
                                       {t: {v: k for k, v in tcode_p2.items()}.get(c) for t, c in tcode_p.items()})
    r_p = M.fit_team_ratings(M.team_match_table(pm_p, fx_p), sorted(tm_p["id"].astype(int)), prior_p)

    dispersion = M.fit_dispersion(pd.concat([pm_p, pm_c]))
    log("dispersion:", dispersion)

    # ---------------- backtest + calibration -----------------
    # Last season is replayed week by week (only past data at each step). Its raw
    # probabilities are graded, and a per-stat calibration is fitted on them; that
    # calibration is then applied to this season and graded out of sample.
    import backtest as B
    stale = False
    if os.path.exists(BACKTEST_OUT):
        with open(BACKTEST_OUT) as fh:
            graded = {m["m"] for m in json.load(fh).get("markets", [])}
        # re-run once last season's shots/fouls arrive (first live run caches them)
        for s, col in (("shots", "shots_v"), ("sot", "sot_v"), ("fouls", "fouls_v")):
            if pm_p[col].notna().any() and s not in graded:
                stale = True
    if args.backtest or stale or not os.path.exists(BACKTEST_OUT):
        log("backtest (last season walk-forward) ...")
        bt, bt_df = B.run(pm_p, pm_p2, fx_p, fx_p2, pl_p, r_p2, tcode_p, tcode_p2, dispersion, PREV, return_df=True)
        bt["platt"] = {m: M.fit_platt(g.p, g.y) for m, g in bt_df.groupby("m")}
        os.makedirs(CACHE, exist_ok=True)
        with open(BACKTEST_OUT, "w") as fh:
            json.dump(bt, fh)
    else:
        with open(BACKTEST_OUT) as fh:
            bt = json.load(fh)
    platt = bt.get("platt", {})
    log("calibration:", platt)

    data_through = pm_c["kickoff"].max()
    ctx = M.build_context(pm_c, pm_p, fx_c, fx_p, pl_c, cutoff=now, prev_team_ratings=r_p,
                          team_code_cur=tcode_c, team_code_prev=tcode_p)
    upcoming = fx_c[(fx_c["kickoff_time"] > data_through) & fx_c["team_h"].notna()].sort_values("kickoff_time")
    next_gw = int(upcoming[upcoming["kickoff_time"] > now - pd.Timedelta(hours=3)]["event"].min()) \
        if len(upcoming) else None
    proj_fx = fx_c[fx_c["event"] == next_gw] if next_gw else fx_c.iloc[0:0]

    short_c = dict(zip(tm_c["id"].astype(int), tm_c["short_name"]))
    fixtures_out, proj_rows = [], []
    for _, f in proj_fx.iterrows():
        team_m, rows = M.project_fixture(ctx, f, pl_c, dispersion)
        rows = M.calibrate_rows(rows, platt)
        for r in rows:
            r["fixture"] = int(f["id"])
        proj_rows += rows
        mk = None
        if len(odds_up):
            o = odds_up[(odds_up.home == f.team_h) & (odds_up.away == f.team_a)]
            if len(o):
                o = o.iloc[0]
                inv = [1 / o.odds_h, 1 / o.odds_d, 1 / o.odds_a]
                tot = sum(inv)
                mk = {"src": o.get("odds_src"), "odds": [clean(o.odds_h), clean(o.odds_d), clean(o.odds_a)],
                      "p": [round(x / tot, 4) for x in inv], "vig": round(tot - 1, 4)}
                if pd.notna(o.get("odds_o25")) and pd.notna(o.get("odds_u25")):
                    io, iu = 1 / o.odds_o25, 1 / o.odds_u25
                    mk["o25_odds"] = [clean(o.odds_o25), clean(o.odds_u25)]
                    mk["p_o25"] = round(io / (io + iu), 4)
        fixtures_out.append({"id": int(f["id"]), "gw": next_gw, "kickoff": f["kickoff_time"].isoformat(),
                             "h": int(f.team_h), "a": int(f.team_a), "model": team_m, "market": mk})

    # ---------------- team table -----------------
    tm_tab_c = M.team_match_table(pm_c, fx_c)
    teams_out = []
    for _, t in tm_c.iterrows():
        tid = int(t["id"])
        d = tm_tab_c[tm_tab_c.team == tid]
        teams_out.append({
            "id": tid, "code": int(t["code"]), "name": t["name"], "short": t["short_name"],
            "att": round(ctx["ratings"]["att"][tid], 3), "def": round(ctx["ratings"]["def"][tid], 3),
            "mp": int(len(d)), "gf": clean(d.gf.sum()), "ga": clean(d.ga.sum()),
            "xgf": clean(d.xgf.sum()), "xga": clean(d.xga.sum()),
            "w": int((d.gf > d.ga).sum()), "d": int((d.gf == d.ga).sum()), "l": int((d.gf < d.ga).sum()),
            "opp_factor": {pos: {s: round(v, 3) for s, v in ctx["oppf"][tid][pos].items()} for pos in M.POSITIONS},
        })

    # ---------------- players -----------------
    players_out = []
    for _, p in pl_c.iterrows():
        code = int(p["code"])
        rates = ctx["rates"].get(code, {})
        players_out.append({
            "id": int(p["id"]), "code": code, "web": p["web_name"],
            "name": f"{p['first_name']} {p['second_name']}", "team": int(p["team"]),
            "pos": S.POS[int(p["element_type"])], "status": p.get("status"),
            "chance": clean(p.get("chance_of_playing_next_round")), "news": p.get("news") if isinstance(p.get("news"), str) else "",
            "news_added": clean(p.get("news_added")),
            "pens": clean(p.get("penalties_order")), "corners": clean(p.get("corners_and_indirect_freekicks_order")),
            "fks": clean(p.get("direct_freekicks_order")),
            "price": clean(p["now_cost"] / 10), "sel": clean(pd.to_numeric(p.get("selected_by_percent"), errors="coerce")),
            "ep_next": clean(pd.to_numeric(p.get("ep_next"), errors="coerce")), "fpl_pts": clean(p.get("total_points")),
            "form": clean(pd.to_numeric(p.get("form"), errors="coerce")),
            "mins_model": ctx["mins"].get(code),
            "rates": {s: {k: clean(v) for k, v in r.items()} for s, r in rates.items()},
        })

    # ---------------- game logs (whole picture: this season + last season) -----------------
    keep_codes = set(pl_c["code"].astype(int))
    short_p = dict(zip(tm_p["id"].astype(int), tm_p["short_name"]))
    logs = []
    for d, sh, fx in ((pm_c, short_c, fx_c), (pm_p[pm_p.code.isin(keep_codes)], short_p, fx_p)):
        d = d.drop(columns=["team_h_score", "team_a_score"], errors="ignore").copy()
        d["team_s"] = d["team"].map(sh)
        d["opp_s"] = d["opp"].map(sh)
        sc = fx.set_index("id")[["team_h_score", "team_a_score"]]
        d = d.join(sc, on="fixture")
        d["score"] = np.where(d.was_home, d.team_h_score.astype("Int64").astype(str) + "-" + d.team_a_score.astype("Int64").astype(str),
                              d.team_a_score.astype("Int64").astype(str) + "-" + d.team_h_score.astype("Int64").astype(str))
        d["kickoff"] = pd.to_datetime(d["kickoff"], utc=True).dt.strftime("%Y-%m-%d")
        for c in LOG_COLS:
            if c not in d.columns:
                d[c] = np.nan
        logs += [[clean(v) for v in row] for row in d[LOG_COLS].itertuples(index=False, name=None)]

    # ---------------- team match stats (football-data) + referees -----------------
    team_stats = []
    for fd, season in ((fd_c, CUR), (fd_p, PREV)):
        if fd is None or not len(fd):
            continue
        sh = short_c if season == CUR else short_p
        for _, r in fd.iterrows():
            team_stats.append({k: clean(r.get(k)) for k in
                               ["date", "h_shots", "a_shots", "h_sot", "a_sot", "h_fouls", "a_fouls", "h_corners",
                                "a_corners", "h_yellow", "a_yellow", "h_red", "a_red", "h_goals", "a_goals", "referee"]}
                              | {"season": season, "home": sh.get(int(r.home)) if pd.notna(r.home) else r.home_name,
                                 "away": sh.get(int(r.away)) if pd.notna(r.away) else r.away_name})
    refs = []
    fd_all = pd.concat([x for x in (fd_c, fd_p) if x is not None and len(x)], ignore_index=True)
    if len(fd_all) and "referee" in fd_all:
        fd_all["fouls"] = fd_all.h_fouls + fd_all.a_fouls
        fd_all["yellows"] = fd_all.h_yellow + fd_all.a_yellow
        fd_all["reds"] = fd_all.h_red + fd_all.a_red
        for ref, g in fd_all.groupby("referee"):
            refs.append({"ref": ref, "matches": int(len(g)), "fouls_pm": clean(g.fouls.mean()),
                         "yellows_pm": clean(g.yellows.mean()), "reds": int(g.reds.sum()),
                         "seasons": sorted(g.season.unique().tolist())})
        refs.sort(key=lambda r: -r["matches"])

    # ---------------- head-to-head fixtures (both seasons, by short name) -----------------
    h2h = []
    for fx, sh, season in ((fx_c, short_c, CUR), (fx_p, short_p, PREV)):
        fin = fx[fx["finished"].astype(str).str.lower().isin(["true", "1"])].dropna(subset=["team_h_score"])
        for _, f in fin.iterrows():
            h2h.append([season, f.kickoff_time.strftime("%Y-%m-%d"), sh[int(f.team_h)], sh[int(f.team_a)],
                        int(f.team_h_score), int(f.team_a_score), int(f["id"])])

    live_raw = B.run(pm_c, pm_p, fx_c, fx_p, pl_c, r_p, tcode_c, tcode_p, dispersion, CUR, min_gw=3,
                     cutoff_max=data_through + pd.Timedelta(seconds=1))
    live = B.run(pm_c, pm_p, fx_c, fx_p, pl_c, r_p, tcode_c, tcode_p, dispersion, CUR, min_gw=3,
                 cutoff_max=data_through + pd.Timedelta(seconds=1), calib=platt)

    has = {s: bool(pm_c[c].notna().any()) if c in pm_c else False
           for s, c in [("shots", "shots"), ("sot", "sot"), ("fouls", "fouls"), ("key_passes", "key_passes")]}
    out = {
        "meta": {
            "season": CUR, "built_at": now.isoformat(), "data_through": clean(data_through),
            "next_gw": next_gw, "sources": sources, "health": health, "dispersion": dispersion,
            "has": has, "ratings_mu": {"home": round(ctx["ratings"]["mu_home"], 3), "away": round(ctx["ratings"]["mu_away"], 3)},
            "news_as_of": clean(pd.to_datetime(pl_c["news_added"], utc=True, errors="coerce").max()),
            "stats": {s: {"label": c["label"], "lines": c["lines"], "adj": c["adj"]} for s, c in M.STATS.items()},
        },
        "teams": teams_out, "players": players_out,
        "fixtures": fixtures_out, "projections": proj_rows,
        "schedule": [{"id": int(f.id), "gw": clean(f.event), "kickoff": clean(f.kickoff_time), "h": int(f.team_h),
                      "a": int(f.team_a), "hs": clean(f.team_h_score), "as": clean(f.team_a_score),
                      "finished": str(f.finished).lower() in ("true", "1") and f.kickoff_time <= data_through}
                     for f in fx_c.dropna(subset=["team_h"]).itertuples()],
        "log_cols": LOG_COLS, "logs": logs,
        "team_stats": team_stats, "referees": refs, "h2h": h2h,
        "prev_teams": {int(k): v for k, v in short_p.items()},
        "backtest": bt, "live_grade": live, "live_grade_raw": live_raw,
    }
    with open(OUT, "w") as fh:
        json.dump(deep_clean(out), fh, separators=(",", ":"), allow_nan=False)
    log(f"wrote {OUT}: {os.path.getsize(OUT)/1e6:.2f} MB, {len(players_out)} players, {len(logs)} log rows, "
        f"{len(proj_rows)} projections for GW{next_gw}")


if __name__ == "__main__":
    main()
