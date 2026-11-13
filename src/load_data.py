"""Load Cricsheet IPL ball-by-ball data (data/ipl_json.zip) and turn it into
state sequences for the two Markov chains used in the project.

  Innings level (our Iteration 1 topic): one state per innings of a top-order batter,
      F = 0-9 runs, S = 10-29, B = 30+, and the absorbing state D = not in his team's
      next match ("dropped"). Encoded F=0, S=1, B=2, D=3.
  Ball level (the course handout's version): one state per ball faced,
      the batter's runs off that ball (0, 1, 2, 3, 4, 6; the 74 five-run balls are put
      with 4), and the absorbing state Out. Encoded 0,1,2,3,4,6 -> 0..5, Out = 6.

Run  `python src/load_data.py`  to print the data description and make the figures.
"""
import collections
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ipl_json.zip"

FAIL_MAX, START_MAX = 9, 29          # F: 0-9, S: 10-29, B: 30+
TOP_ORDER = 4                        # batting positions 1-4
MIN_INNINGS = 40                     # batters with at least this many top-order innings
INNINGS_STATES = ["F", "S", "B", "D"]
BALL_STATES = ["0", "1", "2", "3", "4", "6", "Out"]
NOT_DISMISSALS = {"retired hurt", "retired not out"}


def load_matches(path=DATA_PATH):
    with zipfile.ZipFile(path) as z:
        matches = []
        for name in z.namelist():
            if name.endswith(".json"):
                m = json.loads(z.read(name))
                m["_id"] = name[:-5]
                matches.append(m)
    matches.sort(key=lambda m: m["info"]["dates"][0])
    return matches


def _deliveries(m):
    for inn_no, inn in enumerate(m["innings"]):
        if inn.get("super_over"):
            continue
        for over in inn["overs"]:
            for d in over["deliveries"]:
                yield inn_no, inn["team"], d


def build_appearances(matches):
    """One row per player per team per match: whether he batted, position, runs, balls."""
    rows = []
    for m in matches:
        info, reg = m["info"], m["info"]["registry"]["people"]
        date = info["dates"][0]
        runs, balls, position, bat_team = collections.Counter(), collections.Counter(), {}, {}
        orders = collections.defaultdict(list)
        for inn_no, team, d in _deliveries(m):
            for p in (d["batter"], d["non_striker"]):
                if p not in orders[inn_no]:
                    orders[inn_no].append(p)
                    position[p], bat_team[p] = len(orders[inn_no]), team
            runs[d["batter"]] += d["runs"]["batter"]
            if "wides" not in d.get("extras", {}):
                balls[d["batter"]] += 1
        for team, players in info["players"].items():
            for p in players:
                batted = bat_team.get(p) == team
                rows.append(dict(match_id=m["_id"], date=pd.Timestamp(date), season=int(date[:4]),
                                 team=team, player_id=reg[p], player=p, batted=batted,
                                 position=position[p] if batted else np.nan,
                                 runs=runs[p] if batted else np.nan,
                                 balls=balls[p] if batted else np.nan))
    apps = pd.DataFrame(rows)
    games = (apps[["season", "team", "match_id", "date"]].drop_duplicates()
             .sort_values(["season", "team", "date"]))
    games["game_no"] = games.groupby(["season", "team"]).cumcount()
    return apps.merge(games[["season", "team", "match_id", "game_no"]], on=["season", "team", "match_id"])


def innings_state(r):
    return 0 if r <= FAIL_MAX else (1 if r <= START_MAX else 2)


def innings_level(apps, min_innings=MIN_INNINGS, season_range=None, pool=None):
    """Returns (top, chains, log).
    top:    one row per top-order innings of a pool batter, with its state.
    chains: {player_id: [chain, ...]}; a chain is a list of consecutive linked states and
            ends in 3 (D) if the batter was dropped, otherwise it is censored.
    Transitions are only between a team's consecutive matches in the same season:
      next match, batted top-4      -> that innings' state
      next match, not in the team   -> D
      next match, in team but no top-4 innings (did not bat, moved down) -> sequence break
      no next match (season over)   -> censored"""
    a = apps if season_range is None else apps[apps.season.between(*season_range)]
    top = a[a.batted & (a.position <= TOP_ORDER)].copy()
    top["state"] = top.runs.apply(innings_state)
    if pool is None:
        counts = top.player_id.value_counts()
        pool = counts[counts >= min_innings].index
    top = top[top.player_id.isin(pool)].sort_values(["player_id", "season", "team", "game_no"])
    selected = set(zip(a.player_id, a.season, a.team, a.game_no))
    top_state = {(r.player_id, r.season, r.team, r.game_no): r.state for r in top.itertuples()}
    last_game = a.groupby(["season", "team"]).game_no.max().to_dict()

    chains = collections.defaultdict(list)
    ends = collections.Counter()
    for pid, df in top.groupby("player_id"):
        cur = []
        for r in df.itertuples():
            if not cur:
                cur = [r.state]
            nxt = (pid, r.season, r.team, r.game_no + 1)
            if r.game_no + 1 > last_game[(r.season, r.team)]:
                chains[pid].append(cur); cur = []; ends["season over (censored)"] += 1
            elif nxt not in selected:
                chains[pid].append(cur + [3]); cur = []; ends["dropped (absorbed)"] += 1
            elif nxt in top_state:
                cur = cur + [top_state[nxt]]          # the next row of df continues this chain
            else:
                chains[pid].append(cur); cur = []; ends["played, no top-4 innings (break)"] += 1
        if cur:
            chains[pid].append(cur)
    log = {"pool_batters": top.player_id.nunique(), "pool_innings": len(top), **ends}
    return top, dict(chains), log


def ball_state(runs):
    return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5}.get(runs, 4 if runs < 6 else 5)


def ball_level(matches, pool_ids, top_positions):
    """One sequence per (match, innings, batter) for pool batters batting in the top 4.
    Wides are not balls faced and are skipped (a stumping or run-out on a wide still ends the
    innings with Out). No-balls are balls faced and keep the batter's runs. A dismissal of the
    batter (as striker or non-striker) appends Out = 6. Retired hurt ends the sequence without
    absorption (censored); if he returns, a new sequence starts. Not out at the end of the
    team's innings is censored."""
    seqs, seasons, log = [], [], collections.Counter()
    for m in matches:
        reg = m["info"]["registry"]["people"]
        season = int(m["info"]["dates"][0][:4])
        current = {}
        def close(name, absorbed):
            s = current.pop(name, None)
            if s is not None:
                seqs.append(np.array(s + ([6] if absorbed else []), dtype=int))
                seasons.append(season)
                log["out (absorbed)" if absorbed else "not out / retired hurt (censored)"] += 1
        inn_prev = None
        for inn_no, team, d in _deliveries(m):
            if inn_no != inn_prev:
                for name in list(current):
                    close(name, False)
                inn_prev = inn_no
            ex = d.get("extras", {})
            striker, non = d["batter"], d["non_striker"]
            for p in (striker, non):
                if p not in current and reg[p] in pool_ids and \
                        top_positions.get((m["_id"], inn_no, reg[p]), 99) <= TOP_ORDER:
                    current[p] = []
            out_now = {w["player_out"]: w["kind"] for w in d.get("wickets", [])}
            if "wides" in ex:
                log["wides skipped"] += 1
            elif striker in current:
                if striker in out_now and out_now[striker] not in NOT_DISMISSALS:
                    pass                                  # Out is appended by close() below
                else:
                    current[striker].append(ball_state(d["runs"]["batter"]))
                log["no-balls kept as balls faced"] += "noballs" in ex
            for p, kind in out_now.items():
                if p in current:
                    close(p, kind not in NOT_DISMISSALS)
        for name in list(current):
            close(name, False)
    log["sequences"] = len(seqs)
    log["balls faced"] = int(sum((s < 6).sum() for s in seqs))
    return seqs, np.array(seasons), dict(log)


def batting_positions(matches):
    idx = {}
    for m in matches:
        reg = m["info"]["registry"]["people"]
        orders = collections.defaultdict(list)
        for inn_no, team, d in _deliveries(m):
            for p in (d["batter"], d["non_striker"]):
                if p not in orders[inn_no]:
                    orders[inn_no].append(p)
                    idx[(m["_id"], inn_no, reg[p])] = len(orders[inn_no])
    return idx


def load_all(path=DATA_PATH):
    """Everything the other scripts need, in one call."""
    matches = load_matches(path)
    apps = build_appearances(matches)
    top, chains, inn_log = innings_level(apps)
    seqs, seasons, ball_log = ball_level(matches, set(top.player_id), batting_positions(matches))
    return dict(matches=matches, apps=apps, top=top, chains=chains, inn_log=inn_log,
                ball_seqs=seqs, ball_seasons=seasons, ball_log=ball_log)


def flatten_chains(chains):
    return [np.array(c, dtype=int) for pid in chains for c in chains[pid]]


# ----------------------------------------------------------------------------
# Iteration 2: describe the data
# ----------------------------------------------------------------------------
def describe():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from report_utils import FIG, Macros, tex_table, write_tex

    d = load_all()
    matches, apps, top = d["matches"], d["apps"], d["top"]
    print(f"{len(matches)} matches, {matches[0]['info']['dates'][0]} to {matches[-1]['info']['dates'][0]}")
    print(f"appearances table: {apps.shape[0]:,} rows x {apps.shape[1]} columns")
    print(apps.dtypes.to_string())
    print("missing values per column:\n", apps.isna().sum().to_string())
    print("innings-level log:", d["inn_log"])
    print("ball-level log:", d["ball_log"])

    # variables table
    meaning = {"match_id": "Cricsheet match identifier", "date": "match date",
               "season": "IPL season (year of the match date)", "team": "franchise name",
               "player_id": "Cricsheet registry id (unique per person)", "player": "player name",
               "batted": "did he bat in this match?", "position": "batting position (order at the crease)",
               "runs": "runs scored by the batter", "balls": "balls faced (wides excluded)",
               "game_no": "index of this match within the team's season"}
    rows = [[f"\\texttt{{{c.replace('_', chr(92) + '_')}}}", f"\\texttt{{{apps[c].dtype}}}",
             f"{int(apps[c].isna().sum()):,}".replace(",", "{,}"), meaning[c]] for c in apps.columns]
    write_tex("data_variables.tex", tex_table(["Column", "Type", "Missing", "Meaning"], rows,
              align="llrl", caption=r"The appearances table built by \texttt{load\_data.py} "
              r"(one row per player per team per match).", label="tab:variables"))

    m = Macros("data_numbers.tex")
    fmt = lambda v: f"{int(v):,}".replace(",", "{,}")
    m["nMatches"] = fmt(len(matches))
    m["firstDate"] = matches[0]["info"]["dates"][0]
    m["lastDate"] = matches[-1]["info"]["dates"][0]
    m["nSeasons"] = str(apps.season.nunique())
    m["nApps"] = fmt(len(apps))
    m["nCols"] = str(apps.shape[1])
    m["nPlayers"] = fmt(apps.player_id.nunique())
    m["nDidNotBat"] = fmt(apps.runs.isna().sum())
    m["nPool"] = fmt(d["inn_log"]["pool_batters"])
    m["nPoolInnings"] = fmt(d["inn_log"]["pool_innings"])
    m["nDropped"] = fmt(d["inn_log"].get("dropped (absorbed)", 0))
    m["nCensored"] = fmt(d["inn_log"].get("season over (censored)", 0))
    m["nBreaks"] = fmt(d["inn_log"].get("played, no top-4 innings (break)", 0))
    m["nBallSeqs"] = fmt(d["ball_log"]["sequences"])
    m["nBalls"] = fmt(d["ball_log"]["balls faced"])
    m["nBallOut"] = fmt(d["ball_log"].get("out (absorbed)", 0))
    m["nBallNotOut"] = fmt(d["ball_log"].get("not out / retired hurt (censored)", 0))
    m["nWides"] = fmt(d["ball_log"].get("wides skipped", 0))
    m["nNoBalls"] = fmt(d["ball_log"].get("no-balls kept as balls faced", 0))
    state_counts = top.state.value_counts().reindex([0, 1, 2]).values
    for s, c in zip("FSB", state_counts):
        m[f"nState{s}"] = fmt(c)
    m.save()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.hist(top.runs, bins=np.arange(0, 131, 5), color="#9ecae1", edgecolor="white")
    for v, lab in [(FAIL_MAX + 0.5, "F | S"), (START_MAX + 0.5, "S | B")]:
        ax.axvline(v, color="crimson", lw=1.2, ls="--")
        ax.text(v + 1, ax.get_ylim()[1] * 0.92, lab, color="crimson")
    ax.set_xlabel("runs in the innings"); ax.set_ylabel("number of innings")
    ax.set_title(f"Top-order innings of the {d['inn_log']['pool_batters']} pool batters")
    ax = axes[1]
    per_season = top.groupby("season").size()
    ax.bar(per_season.index, per_season.values, color="#a1d99b")
    ax.set_xticks(per_season.index[::3])
    ax.set_xlabel("season"); ax.set_ylabel("innings in the pool"); ax.set_title("Innings per season")
    plt.tight_layout(); plt.savefig(FIG / "data_innings.png", dpi=150); plt.close()

    balls = np.concatenate(d["ball_seqs"])
    cnt = np.bincount(balls, minlength=7)
    plt.figure(figsize=(6, 3.2))
    plt.bar(BALL_STATES, cnt / cnt.sum(), color=["#c6dbef"] * 6 + ["#fc9272"])
    plt.ylabel("share of states"); plt.xlabel("ball outcome (runs to the batter) or Out")
    plt.title("Ball-level states of the pool batters")
    plt.tight_layout(); plt.savefig(FIG / "data_balls.png", dpi=150); plt.close()
    print("figures written to", FIG)


if __name__ == "__main__":
    describe()
