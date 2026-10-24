"""
Walk-forward backtest: for every gameweek, rebuild the model using only matches
played before that gameweek's first kickoff, project every player who ended up
starting, and grade each over/under line against what actually happened.

Baseline = the player's raw hit rate in his starts so far this season
(what most prop apps show), falling back to the position average under 3 starts.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import model as M

BINS = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]


def _brier(p, y):
    p, y = np.asarray(p), np.asarray(y)
    return float(np.mean((p - y) ** 2))


def _logloss(p, y):
    p = np.clip(np.asarray(p), 0.02, 0.98)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run(pm, pm_prior, fx, fx_prior, players, prior_ratings, tcode, tcode_prior, dispersion, season,
        min_gw=6, cutoff_max=None, calib=None, return_df=False):
    fin = fx[fx["finished"].astype(str).str.lower().isin(["true", "1"])].dropna(subset=["team_h_score", "event"])
    if cutoff_max is not None:
        fin = fin[fin["kickoff_time"] <= cutoff_max]
    gws = sorted(int(g) for g in fin["event"].unique() if g >= min_gw)
    actual = pm.set_index(["code", "fixture"])
    recs, trecs = [], []
    for gw in gws:
        gfx = fin[fin["event"] == gw]
        cutoff = gfx["kickoff_time"].min()
        ctx = M.build_context(pm, pm_prior, fx, fx_prior, players, cutoff, prior_ratings, tcode, tcode_prior)
        before = pm[(pm["kickoff"] < cutoff) & (pm["starts"] >= 1)]
        before = before.assign(pos=before["code"].map(ctx["pos_of"]))
        by_code = dict(tuple(before.groupby("code")))
        by_pos = dict(tuple(before.groupby("pos")))
        empty = before.iloc[0:0]
        for _, f in gfx.iterrows():
            team_m, rows = M.project_fixture(ctx, f, players, dispersion, use_availability=False)
            rows = M.calibrate_rows(rows, calib)
            hs, as_ = int(f.team_h_score), int(f.team_a_score)
            trecs.append({"gw": gw, "ph": team_m["p_home"], "pd": team_m["p_draw"], "pa": team_m["p_away"],
                          "o25": team_m["o25"], "btts": team_m["btts"],
                          "res": 0 if hs > as_ else (1 if hs == as_ else 2), "y_o25": int(hs + as_ >= 3),
                          "y_btts": int(hs > 0 and as_ > 0)})
            for r in rows:
                key = (r["code"], int(f["id"]))
                if key not in actual.index:
                    continue
                a = actual.loc[key]
                if isinstance(a, pd.DataFrame):
                    a = a.iloc[0]
                if a["starts"] < 1:
                    continue
                hist = by_code.get(r["code"], empty)
                posd = by_pos.get(r["pos"], empty)
                for s, mk in r["m"].items():
                    y_val = a.get(f"{s}_o")
                    if y_val is None or (isinstance(y_val, float) and math.isnan(y_val)):
                        continue
                    hv = hist[f"{s}_o"].dropna()
                    pv = posd[f"{s}_o"].dropna()
                    for L, p_start, _p_play in mk["lines"]:
                        if len(hv) >= 3:
                            base = float((hv >= L).mean())
                        elif len(pv):
                            base = float((pv >= L).mean())
                        else:
                            continue
                        recs.append((s, L, p_start, base, int(y_val >= L), gw))
    if not recs:
        res = {"season": season, "gws": [], "markets": [], "calibration": {}, "team": None, "n": 0}
        return (res, pd.DataFrame(columns=["m", "line", "p", "base", "y", "gw"])) if return_df else res
    df = pd.DataFrame(recs, columns=["m", "line", "p", "base", "y", "gw"])
    markets = []
    for (m, L), g in df.groupby(["m", "line"]):
        if len(g) < 50:
            continue
        bm, bb = _brier(g.p, g.y), _brier(g.base, g.y)
        markets.append({"m": m, "line": L, "n": int(len(g)), "hit_rate": round(float(g.y.mean()), 4),
                        "avg_pred": round(float(g.p.mean()), 4),
                        "brier_model": round(bm, 5), "brier_base": round(bb, 5),
                        "ll_model": round(_logloss(g.p, g.y), 4), "ll_base": round(_logloss(g.base, g.y), 4),
                        "skill": round(1 - bm / bb, 4) if bb > 0 else None})
    calib = {}
    for m, g in df.groupby("m"):
        g = g.assign(b=pd.cut(g.p, BINS, right=False))
        rows = []
        for b, gg in g.groupby("b", observed=True):
            if len(gg) >= 30:
                rows.append([round(float(gg.p.mean()), 4), round(float(gg.y.mean()), 4), int(len(gg))])
        calib[m] = rows
    # baseline calibration (same bins on the hit-rate number) for the side-by-side chart
    calib_base = {}
    for m, g in df.groupby("m"):
        g = g.assign(b=pd.cut(g.base, BINS, right=False))
        calib_base[m] = [[round(float(gg.base.mean()), 4), round(float(gg.y.mean()), 4), int(len(gg))]
                         for b, gg in g.groupby("b", observed=True) if len(gg) >= 30]

    t = pd.DataFrame(trecs)
    team = None
    if len(t):
        P = t[["ph", "pd", "pa"]].to_numpy()
        Y = np.eye(3)[t.res.to_numpy()]
        # baseline: league-wide frequencies from the backtest period
        freq = Y.mean(axis=0)
        team = {
            "n": int(len(t)),
            "brier_1x2_model": round(float(np.mean(np.sum((P - Y) ** 2, axis=1))), 4),
            "brier_1x2_base": round(float(np.mean(np.sum((freq - Y) ** 2, axis=1))), 4),
            "brier_o25_model": round(_brier(t.o25, t.y_o25), 4),
            "brier_o25_base": round(_brier(np.full(len(t), t.y_o25.mean()), t.y_o25), 4),
            "brier_btts_model": round(_brier(t.btts, t.y_btts), 4),
            "brier_btts_base": round(_brier(np.full(len(t), t.y_btts.mean()), t.y_btts), 4),
            "fav_hit": round(float((P.argmax(axis=1) == t.res.to_numpy()).mean()), 4),
            "calib_home": [[round(float(g.ph.mean()), 3), round(float((g.res == 0).mean()), 3), int(len(g))]
                           for _, g in t.assign(b=pd.cut(t.ph, [0, .2, .3, .4, .5, .6, .7, 1.01])).groupby("b", observed=True)
                           if len(g) >= 10],
        }
    res = {"season": season, "gws": [gws[0], gws[-1]], "n": int(len(df)), "markets": markets,
            "calibration": calib, "calibration_base": calib_base, "team": team,
            "note": "Graded on players who started. Model numbers are 'if he starts' probabilities; "
                    "baseline is his raw hit rate in previous starts this season.",
           "calibrated": bool(calib)}
    return (res, df) if return_df else res
