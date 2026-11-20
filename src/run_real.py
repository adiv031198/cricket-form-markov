"""Iteration 3: run the from-scratch implementation on the real Cricsheet IPL data,
compare every result with a library routine, check the assumptions, answer the
Iteration 3 math questions 4-5, and build the verification table.

Run after small_example.py and check_simulation.py:  python src/run_real.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from load_data import (drop_events, BALL_STATES, INNINGS_STATES, MIN_INNINGS, TOP_ORDER, flatten_chains, innings_level,
                       load_all)
from methods import (my_pooling_statistic, my_censoring_statistic, my_absorption_probs, my_absorption_time_variance, my_absorption_times,
                     my_count_transitions, my_fundamental_matrix, my_gauss_solve, my_homogeneity_statistic,
                     my_markov_test_statistic, my_matrix_power, my_row_normalize, my_second_order_P,
                     my_standard_errors, my_steady_state, my_steady_state_system)
from report_utils import (FIG, Macros, log_verification, read_verification, report, tex_diff, tex_num,
                          tex_table, write_tex)


def eig_pi(P):
    """Library check only: eigenvector of P^T for the eigenvalue closest to 1, scaled to sum 1.
    Eigenvectors are defined only up to a scalar (and sign), so we divide by the sum."""
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def fmt_int(v):
    return f"{int(v):,}".replace(",", "{,}")


def analyse(name, seqs, labels, k, m, ver, tag, chain):
    """Full analysis of one chain: k transient states, absorbing state index k."""
    out = {}
    n3 = my_count_transitions(seqs, k)               # transient -> transient only
    n_full = my_count_transitions(seqs, k, k + 1)    # transient -> transient or absorbing
    P = my_row_normalize(n3)
    P_full = my_row_normalize(n_full)
    se = my_standard_errors(P, n3.sum(axis=1))
    pi = my_steady_state(P)
    a_pi, r_pi = report(f"{name}: pi vs eig", pi, eig_pi(P))
    A, e = my_steady_state_system(P)
    a_ge, _ = report(f"{name}: Gauss vs numpy.solve", my_gauss_solve(A, e), np.linalg.solve(A, e))
    a_pw, _ = report(f"{name}: P^5 vs matrix_power", my_matrix_power(P, 5), np.linalg.matrix_power(P, 5))
    Q, R = P_full[:, :k], P_full[:, k:]
    N = my_fundamental_matrix(Q)
    a_N, _ = report(f"{name}: N vs inv(I - Q)", N, np.linalg.inv(np.eye(k) - Q))
    t = my_absorption_times(N)
    var = my_absorption_time_variance(N, t)
    B = my_absorption_probs(N, R)
    lam = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]      # library, only to report |lambda_2|
    out.update(n3=n3, n_full=n_full, P=P, P_full=P_full, se=se, pi=pi, N=N, t=t, var=var, B=B, lam2=lam[1])
    print(f"{name}: P_full =\n", P_full.round(3))
    print(f"{name}: pi = {pi.round(4)}, |lambda_2| = {lam[1]:.4f}")
    print(f"{name}: t = N1 = {t.round(2)}, sd = {np.sqrt(var).round(2)}, B = {B.ravel().round(6)}")
    ver += [[f"Real data, {chain}", r"$\pi$", r"\texttt{numpy.linalg.eig}", tex_diff(a_pi), a_pi < 1e-10],
            [f"Real data, {chain}", r"$\pi$ linear system", r"\texttt{numpy.linalg.solve}", tex_diff(a_ge), a_ge < 1e-10],
            [f"Real data, {chain}", r"$\hat P^5$", r"\texttt{numpy.linalg.matrix\_power}", tex_diff(a_pw), a_pw < 1e-10],
            [f"Real data, {chain}", r"$N=(I-Q)^{-1}$", r"\texttt{numpy.linalg.inv}", tex_diff(a_N), a_N < 1e-10]]
    m[f"{tag}PiEigErr"] = tex_num(a_pi)
    m[f"{tag}PiEigRel"] = tex_num(r_pi)
    m[f"{tag}NErr"] = tex_num(a_N)
    m[f"{tag}LamTwo"] = f"{lam[1]:.4f}"

    # transition table with 95% intervals (rows: from; columns: to, incl. absorbing)
    se_full = my_standard_errors(P_full, n_full.sum(axis=1))
    rows = [[f"{labels[i]} ($n_i={fmt_int(n_full[i].sum())}$)"] +
            [f"{P_full[i, j]:.3f} $\\pm$ {1.96 * se_full[i, j]:.3f}" for j in range(k + 1)] for i in range(k)]
    write_tex(f"real_{tag}_P.tex", tex_table(["From"] + [f"to {l}" for l in labels], rows,
              align="l" + "r" * (k + 1),
              caption=f"{name}: estimated transition probabilities with 95\\% intervals "
              r"$\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n_i}$.", label=f"tab:real{tag}P"))
    rows = [[labels[i], f"{pi[i]:.4f}", f"{t[i]:.2f}", f"{np.sqrt(var[i]):.2f}", f"{B[i, 0]:.4f}"] for i in range(k)]
    write_tex(f"real_{tag}_summary.tex", tex_table(
        ["State $i$", r"$\pi_i$", r"$t_i=(N\mathbf 1)_i$", r"$\mathrm{sd}(\tau_i)$", r"$B_i=(NR)_i$"], rows,
        caption=f"{name}: steady state of the chain without absorption, expected steps before absorption, "
        "their standard deviation and the absorption probability.", label=f"tab:real{tag}sum"))

    # Markov test (Iteration 3 Q5)
    x2, df = my_markov_test_statistic(seqs, k)
    p = stats.chi2.sf(x2, df)
    P2 = my_second_order_P(seqs, k)
    out.update(markov=(x2, df, p), P2=P2)
    print(f"{name}: Markov test X2 = {x2:.1f}, df = {df}, p = {p:.3g}")
    m[f"{tag}MarkovX"], m[f"{tag}MarkovDf"], m[f"{tag}MarkovP"] = f"{x2:.1f}", str(df), tex_num(p, 3)
    return out


def second_order_table(name, P2, P, labels, k, tag, target):
    """Pr(X_{t+1} = target | X_t = i, X_{t-1} = h) next to Pr(X_{t+1} = target | X_t = i)."""
    rows = []
    for i in range(k):
        vals = [f"{P2[h, i, target]:.3f}" if not np.isnan(P2[h, i, target]) else "--" for h in range(k)]
        rows.append([labels[i], f"{P[i, target]:.3f}"] + vals)
    write_tex(f"real_{tag}_second.tex", tex_table(
        [r"$X_t$", r"$\Pr(\cdot\mid X_t)$"] + [f"$X_{{t-1}}={l}$" for l in labels[:k]], rows,
        caption=f"{name}, Markov check: probability that the next state is {labels[target]}, given the current "
        "state only (second column) and given the current and the previous state (other columns).",
        label=f"tab:real{tag}second"))


def assumption_rows(name, seqs, seqs_a, seqs_b, k, per_player=None):
    rows = []
    x2, df = my_markov_test_statistic(seqs, k)
    p = stats.chi2.sf(x2, df)
    rows.append([name, "A1 Markov (first order)", rf"$X^2={x2:.1f}$, df $={df}$, $p={tex_num(p, 3)}$", p > 0.05])
    hx2, hdf = my_homogeneity_statistic(my_count_transitions(seqs_a, k), my_count_transitions(seqs_b, k))
    hp = stats.chi2.sf(hx2, hdf)
    rows.append([name, "A2 Time-homogeneous", rf"2008--16 vs 2017--26: $X^2={hx2:.1f}$, df $={hdf}$, $p={tex_num(hp, 3)}$", hp > 0.05])
    P = my_row_normalize(my_count_transitions(seqs, k))
    P2 = my_matrix_power(P, 2)
    rows.append([name, "A3 Regular", rf"$\min_{{ij}}(\hat P^2)_{{ij}} = {P2.min():.4f}$", P2.min() > 0])
    n_i = my_count_transitions(seqs, k).sum(axis=1)
    rows.append([name, "A4 Enough data per row", rf"$\min_i n_i = {fmt_int(n_i.min())}$", n_i.min() >= 100])
    Pf = my_row_normalize(my_count_transitions(seqs, k, k + 1))
    rows.append([name, "A5 Absorption reachable", rf"$\max_i (\hat Q\mathbf 1)_i = {Pf[:, :k].sum(axis=1).max():.3f}$",
                 Pf[:, :k].sum(axis=1).max() < 1])
    if per_player is not None:
        x2, df = my_pooling_statistic([my_count_transitions(s, k) for s in per_player])
        pp = stats.chi2.sf(x2, df)
        rows.append([name, "A6 Same $P$ for every batter", rf"batters $\times$ next state: $X^2={x2:.0f}$, df $={df}$, $p={tex_num(pp, 3)}$", pp > 0.05])
    cx2, cdf, rate = my_censoring_statistic(seqs, k)
    cp = stats.chi2.sf(cx2, cdf)
    rate_txt = ", ".join(f"{v:.3f}" for v in rate)
    rows.append([name, "A7 Non-informative censoring", rf"censoring rate by state ${rate_txt}$: $X^2={cx2:.1f}$, df $={cdf}$, $p={tex_num(cp, 3)}$", cp > 0.05])
    for r in rows:
        print(f"   {r[0]:8s} {r[1]:30s} {r[2]:70s} {'pass' if r[3] else 'FAIL'}")
    return rows


def main():
    d = load_all()
    ver, m = [], Macros("real_numbers.tex")
    apps, top, chains = d["apps"], d["top"], d["chains"]
    inn_seqs = flatten_chains(chains)
    ball_seqs = d["ball_seqs"]

    # cleaning record
    il, bl = d["inn_log"], d["ball_log"]
    n_super = sum(1 for mm in d["matches"] for inn in mm["innings"] if inn.get("super_over"))
    rows = [
        ["Matches in the archive", fmt_int(len(d["matches"]))],
        ["Super-over innings excluded", fmt_int(n_super)],
        ["Player appearances (rows of the appearances table)", fmt_int(len(apps))],
        ["Appearances without an innings (did not bat)", fmt_int(apps.runs.isna().sum())],
        [f"Innings in batting positions 1--{TOP_ORDER}", fmt_int((apps.batted & (apps.position <= TOP_ORDER)).sum())],
        [f"Pool: batters with $\\ge {MIN_INNINGS}$ such innings", fmt_int(il["pool_batters"])],
        ["Innings of the pool batters (innings level)", fmt_int(il["pool_innings"])],
        [r"\quad followed by a drop (absorbed in D)", fmt_int(il.get("dropped (absorbed)", 0))],
        [r"\quad team's last match of the season (censored)", fmt_int(il.get("season over (censored)", 0))],
        [r"\quad next match played but no top-4 innings (sequence break)", fmt_int(il.get("played, no top-4 innings (break)", 0))],
        ["Innings-level sequences (chains)", fmt_int(len(inn_seqs))],
        ["Ball-level sequences (one per innings)", fmt_int(bl["sequences"])],
        [r"\quad balls faced (wides excluded, no-balls kept)", fmt_int(bl["balls faced"])],
        [r"\quad ended by a dismissal (absorbed in Out)", fmt_int(bl.get("out (absorbed)", 0))],
        [r"\quad not out or retired hurt (censored)", fmt_int(bl.get("not out / retired hurt (censored)", 0))],
    ]
    write_tex("real_cleaning.tex", tex_table(["Cleaning step", "Count"], rows, align="lr",
              caption=r"Every cleaning and selection step, as counted by \texttt{run\_real.py}.", label="tab:cleaning"))

    # ---------------------------------------------------------------- innings level
    print("\n=== Innings level ===")
    inn = analyse("Innings level", inn_seqs, INNINGS_STATES, 3, m, ver, "inn", "Chain I")
    second_order_table("Innings level", inn["P2"], inn["P"], INNINGS_STATES, 3, "inn", target=2)
    Pf = inn["P_full"]
    m["innDropF"], m["innDropS"], m["innDropB"] = (f"{Pf[i, 3]:.3f}" for i in range(3))
    m["innDropRatio"] = f"{Pf[0, 3] / Pf[2, 3]:.1f}"
    m["innBigF"], m["innBigS"], m["innBigB"] = (f"{inn['P'][i, 2]:.3f}" for i in range(3))
    m["innTF"], m["innTS"], m["innTB"] = (f"{v:.1f}" for v in inn["t"])
    m["innPiF"], m["innPiS"], m["innPiB"] = (f"{v:.3f}" for v in inn["pi"])
    emp = top.state.value_counts(normalize=True).reindex([0, 1, 2]).values
    m["innEmpF"], m["innEmpS"], m["innEmpB"] = (f"{v:.3f}" for v in emp)

    # ---------------------------------------------------------------- ball level
    print("\n=== Ball level ===")
    ball = analyse("Ball level", ball_seqs, BALL_STATES, 6, m, ver, "ball", "Chain II")
    second_order_table("Ball level", ball["P2"], ball["P"], BALL_STATES, 6, "ball", target=4)
    Pb = ball["P_full"]
    m["ballOutMin"], m["ballOutMax"] = f"{Pb[:, 6].min():.3f}", f"{Pb[:, 6].max():.3f}"
    m["ballOutZero"], m["ballOutSix"] = f"{Pb[0, 6]:.3f}", f"{Pb[5, 6]:.3f}"
    m["ballTZero"], m["ballTOne"], m["ballTSix"] = f"{ball['t'][0]:.1f}", f"{ball['t'][1]:.1f}", f"{ball['t'][5]:.1f}"
    m["ballFourAfterFour"], m["ballFourAfterZero"] = f"{ball['P'][4, 4]:.3f}", f"{ball['P'][0, 4]:.3f}"

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.7]})
    for ax, M, labs, title in [(axes[0], inn["P_full"], INNINGS_STATES, "Innings level"),
                               (axes[1], ball["P_full"], BALL_STATES, "Ball level")]:
        ax.imshow(M, cmap="Blues", vmin=0, vmax=0.6)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > 0.35 else "black")
        ax.set_xticks(range(len(labs)), labs); ax.set_yticks(range(M.shape[0]), labs[:M.shape[0]])
        ax.set_xlabel("next state"); ax.set_ylabel("current state"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(FIG / "real_transition_matrices.png", dpi=150); plt.close()

    # ---------------------------------------------------------------- assumptions on real data
    print("\nAssumption checks on the real data:")
    early, late = innings_level(apps, season_range=(2008, 2016), pool=set(top.player_id)), \
        innings_level(apps, season_range=(2017, 2026), pool=set(top.player_id))
    per_player = [[np.array(c) for c in chains[pid]] for pid in chains]
    rows = assumption_rows("Innings", inn_seqs, flatten_chains(early[1]), flatten_chains(late[1]), 3, per_player)
    seasons = d["ball_seasons"]
    rows += assumption_rows("Ball", ball_seqs, [s for s, y in zip(ball_seqs, seasons) if y <= 2016],
                            [s for s, y in zip(ball_seqs, seasons) if y >= 2017], 6)
    # A8: does "not in the next match" mean dropped? Separate one-match absences from longer ones.
    ev = drop_events(apps, top)
    Pf, nrow = inn["P_full"], inn["n_full"].sum(axis=1)
    all_d = np.array([sum(1 for s, b in ev if s == i) for i in range(3)])
    back = np.array([sum(1 for s, b in ev if s == i and b) for i in range(3)])
    strict = all_d - back                                   # still absent at the following match
    p_strict = strict / (nrow - back)                       # one-match absences are not counted as drops
    se_diff = np.sqrt(p_strict[0] * (1 - p_strict[0]) / (nrow[0] - back[0]) + p_strict[2] * (1 - p_strict[2]) / (nrow[2] - back[2]))
    z8 = (p_strict[0] - p_strict[2]) / se_diff
    share_back = back.sum() / all_d.sum()
    print(f"   A8: {len(ev)} absences, {share_back:.1%} back after one match; strict P(D|F) = {p_strict[0]:.3f}, "
          f"P(D|B) = {p_strict[2]:.3f}, ratio {p_strict[0] / p_strict[2]:.1f}, z = {z8:.1f}")
    rows.append(["Innings", "A8 ``Not selected'' means dropped",
                 rf"{share_back * 100:.0f}\% of absences last one match; without them $\Pr(D\mid F)={p_strict[0]:.3f}$, "
                 rf"$\Pr(D\mid B)={p_strict[2]:.3f}$ ({p_strict[0] / p_strict[2]:.1f}$\times$, $z={z8:.1f}$)", z8 > 3])
    m["aEightBack"] = f"{share_back:.0%}".replace("%", r"\%")
    m["aEightF"], m["aEightB"] = f"{p_strict[0]:.3f}", f"{p_strict[2]:.3f}"
    m["aEightRatio"] = f"{p_strict[0] / p_strict[2]:.1f}"
    m["aEightN"] = fmt_int(len(ev))

    # A7 follow-up: split Chain I censoring into season end and "played, but no top-4 innings next match"
    sel = set(zip(apps.player_id, apps.season, apps.team, apps.game_no))
    tops = {(t.player_id, t.season, t.team, t.game_no) for t in top.itertuples()}
    lastg = apps.groupby(["season", "team"]).game_no.max().to_dict()
    vis, se_end, brk = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in top.itertuples():
        vis[t.state] += 1
        nxt = (t.player_id, t.season, t.team, t.game_no + 1)
        if t.game_no + 1 > lastg[(t.season, t.team)]:
            se_end[t.state] += 1
        elif nxt in sel and nxt not in tops:
            brk[t.state] += 1
    print("   A7 follow-up: season-end rate", (se_end / vis).round(3), " moved down / did not bat rate", (brk / vis).round(3))
    for i, s in enumerate("FSB"):
        m[f"censSeason{s}"], m[f"censBreak{s}"] = f"{se_end[i] / vis[i]:.3f}", f"{brk[i] / vis[i]:.3f}"

    write_tex("assumptions_real.tex", tex_table(
        ["Chain", "Assumption", "Check on the real data", "Pass?"],
        [[r[0], r[1], r[2], r"\checkmark" if r[3] else r"$\times$"] for r in rows],
        align=r"l>{\raggedright\arraybackslash}p{0.25\linewidth}>{\raggedright\arraybackslash}p{0.5\linewidth}c",
        caption="Assumption checks on the real data (same functions as on the simulated data).", label="tab:assreal"))

    # ---------------------------------------------------------------- verification table
    log_verification("real", ver)
    allrows = read_verification(["small", "sim", "real"])
    write_tex("verification_table.tex", tex_table(
        ["Data", "Quantity", "Compared with", "Difference", "Pass?"],
        [[r[0], r[1], r[2], r[3], r"\checkmark" if r[4] else r"$\times$"] for r in allrows], align="lllrc",
        caption=r"Verification table. Every number is produced by \texttt{small\_example.py}, "
        r"\texttt{check\_simulation.py} or \texttt{run\_real.py}.", label="tab:verification"))
    m.save()
    print(f"\nverification rows: {len(allrows)}, all pass: {all(r[4] for r in allrows)}")


if __name__ == "__main__":
    main()
