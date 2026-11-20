"""Iteration 3: run the from-scratch functions on the Iteration 2 simulated data.

  1. Recover the truth (P_hat vs P_TRUE within 3 standard errors) and agree with the
     library (numpy.linalg, check only) at rounding level.
  2. Error against the truth for several sample sizes, on log-log axes.
  3. Absorbing chain: N = (I - Q)^-1 and t = N1 from the data vs the truth vs simulation.
  4. Stress test: a chain that is not Markov; the Markov test must detect it.
  5. Assumption checks on the simulated data (they hold by construction, so they should pass).

Run:  python src/check_simulation.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from methods import (my_pooling_statistic, my_censoring_statistic, my_absorption_probs, my_absorption_times, my_canonical_form, my_count_transitions,
                     my_estimate_P, my_fundamental_matrix, my_homogeneity_statistic, my_markov_test_statistic,
                     my_matrix_power, my_row_normalize, my_second_order_P, my_standard_errors, my_steady_state)
from report_utils import (FIG, Macros, log_verification, report, tex_diff, tex_matrix, tex_num, tex_table,
                          write_tex)
from simulate import (P_OUT, P_TRUE, SEED, absorbing_matrix, simulate_absorbing, simulate_chain,
                      simulate_n_step_fractions, simulate_second_order)

K = 3


def eig_pi(P):
    """Library check only: left eigenvector of P for eigenvalue 1, scaled to sum to 1."""
    vals, vecs = np.linalg.eig(np.asarray(P, float).T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def assumption_checks(x_or_seqs, k, n_full=None, label=""):
    """Returns rows for the assumption-check table and prints them."""
    rows = []
    x2, df = my_markov_test_statistic(x_or_seqs, k)
    p = stats.chi2.sf(x2, df)
    rows.append(["A1 Markov (first order)", rf"$X^2={x2:.1f}$, df $={df}$, $p={p:.3f}$", p > 0.05])
    seqs = x_or_seqs if not np.ndim(x_or_seqs[0]) == 0 else None
    if seqs is None:
        half = len(x_or_seqs) // 2
        n_a, n_b = my_count_transitions(x_or_seqs[:half], k), my_count_transitions(x_or_seqs[half:], k)
    else:
        half = len(seqs) // 2
        n_a, n_b = my_count_transitions(seqs[:half], k), my_count_transitions(seqs[half:], k)
    hx2, hdf = my_homogeneity_statistic(n_a, n_b)
    hp = stats.chi2.sf(hx2, hdf)
    rows.append(["A2 Time-homogeneous", rf"1st vs 2nd half: $X^2={hx2:.1f}$, df $={hdf}$, $p={hp:.3f}$", hp > 0.05])
    P_hat, n_i = my_estimate_P(x_or_seqs, k)
    P2 = my_matrix_power(P_hat, 2)
    rows.append(["A3 Regular (irreducible, aperiodic)", rf"$\min_{{ij}}(\hat P^2)_{{ij}} = {P2.min():.3f}$", P2.min() > 0])
    rows.append(["A4 Enough data in every row", rf"$\min_i n_i = {int(n_i.min()):,}$".replace(",", "{,}"), n_i.min() >= 100])
    if n_full is not None:
        P_full = my_row_normalize(n_full)
        Q, R = P_full[:, :k], P_full[:, k:]
        rows.append(["A5 Absorption reachable", rf"$\max_i (Q\mathbf 1)_i = {Q.sum(axis=1).max():.3f} < 1$",
                     Q.sum(axis=1).max() < 1])
    for r in rows:
        print(f"   {label}{r[0]:38s} {r[1]:60s} {'pass' if r[2] else 'FAIL'}")
    return rows


def main():
    ver, m = [], Macros("sim3_numbers.tex")
    pi_true = my_steady_state(P_TRUE)

    # ------------------------------------------------------------ 0. the handout's starter code, as written
    print("Handout starter code (Project 5):")
    P_true = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.4, 0.5]])
    x = simulate_chain(P_true, T=20000, seed=0)
    P_hat, N_i = my_estimate_P(x, k=3)
    pi = my_steady_state(P_hat)
    vals, vecs = np.linalg.eig(P_hat.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))]); v = v / v.sum()
    report("pi vs eig", pi, v)
    se = np.sqrt(P_true * (1 - P_true) / N_i[:, None])
    print("   max|P_hat - P| / max se =", round(float(np.abs(P_hat - P_true).max() / se.max()), 2), "(< 3)")
    second = my_second_order_P(x, k=3)
    print("   second-order P, rows [h, i, :]:\n", second.round(2))

    # ------------------------------------------------------------ 1. truth and library
    x = simulate_chain(P_TRUE, T=20_000, seed=SEED)
    P_hat, n_i = my_estimate_P(x, K)
    se = my_standard_errors(P_TRUE, n_i)
    z = np.abs(P_hat - P_TRUE).max() / se.max()
    z_each = (np.abs(P_hat - P_TRUE) / se).max()
    pi = my_steady_state(P_hat)
    print("P_hat =\n", P_hat.round(4), "\nrow sums", P_hat.sum(axis=1))
    print(f"max |P_hat - P| / max se = {z:.2f};  max_ij |P_hat_ij - P_ij| / se_ij = {z_each:.2f}  (< 3)")
    a_pi, _ = report("pi vs eig", pi, eig_pi(P_hat))
    a_pow, _ = report("P_hat^3 vs matrix_power", my_matrix_power(P_hat, 3), np.linalg.matrix_power(P_hat, 3))
    report("pi_hat vs true pi", pi, pi_true)
    F = simulate_n_step_fractions(P_TRUE, n=3, n_paths=20_000, seed=SEED)
    a_nstep, _ = report("P_true^3 vs simulated fractions", my_matrix_power(P_TRUE, 3), F)
    P2 = my_second_order_P(x, K)
    print("second-order P(X_t+1 = j | X_t = i, X_t-1 = h), [h, i, :]\n", P2.round(2))
    ver += [["Simulation ($T=20{,}000$)", r"$\hat P$", r"true $P$", tex_diff(z_each, "z"), z_each < 3],
            ["Simulation ($T=20{,}000$)", r"$\pi$ from $\hat P$", r"\texttt{numpy.linalg.eig}", tex_diff(a_pi), a_pi < 1e-10],
            ["Simulation ($T=20{,}000$)", r"$\hat P^3$", r"\texttt{numpy.linalg.matrix\_power}", tex_diff(a_pow), a_pow < 1e-10],
            ["Simulation", r"$P^3$", "fraction of simulated paths", tex_diff(a_nstep), a_nstep < 0.02]]
    m["simThreeZ"] = f"{z_each:.2f}"
    m["simThreePiEig"] = tex_num(a_pi)
    m["simThreePhat"] = tex_matrix(P_hat, 3)

    # second-order table for the control chain
    lab = ["0", "1", "2"]
    rows = [[f"$X_t={lab[i]}$" if h == 0 else "", f"$X_{{t-1}}={lab[h]}$"] + [f"{P2[h, i, j]:.3f}" for j in range(K)]
            for i in range(K) for h in range(K)]
    write_tex("sim3_second_order.tex", tex_table(
        ["", "", r"$\Pr(X_{t+1}=0\mid\cdot)$", r"$\Pr(X_{t+1}=1\mid\cdot)$", r"$\Pr(X_{t+1}=2\mid\cdot)$"], rows,
        align="llrrr", caption=r"Control for the Markov test: second-order estimates on the simulated (Markov) chain. "
        r"Within each block the three rows agree up to sampling noise, as they should.", label="tab:sim3second"))

    # ------------------------------------------------------------ 2. error against T
    sizes = [50, 200, 500, 2000, 5000, 20000]
    reps = 20
    err_P, err_pi, skipped = [], [], 0
    for T in sizes:
        eP, epi = [], []
        for r in range(reps):
            xr = simulate_chain(P_TRUE, T=T, seed=r)
            try:
                Pr, _ = my_estimate_P(xr, K)
            except ValueError:
                skipped += 1
                continue
            eP.append(np.sqrt(((Pr - P_TRUE) ** 2).sum()))
            epi.append(np.abs(my_steady_state(Pr) - pi_true).sum())
        err_P.append(np.mean(eP)); err_pi.append(np.mean(epi))
    slope_P = np.polyfit(np.log(sizes), np.log(err_P), 1)[0]
    slope_pi = np.polyfit(np.log(sizes), np.log(err_pi), 1)[0]
    print("error vs T:", dict(zip(sizes, np.round(err_P, 4))), f"slope {slope_P:.2f};  pi slope {slope_pi:.2f}")
    plt.figure(figsize=(6, 3.6))
    plt.loglog(sizes, err_P, "o-", label=r"$\|\hat P - P\|_F$")
    plt.loglog(sizes, err_pi, "s-", label=r"$\|\hat\pi - \pi\|_1$")
    ref = err_P[0] * (np.array(sizes) / sizes[0]) ** -0.5
    plt.loglog(sizes, ref, "k--", lw=1, label=r"slope $-1/2$")
    plt.xlabel("path length $T$"); plt.ylabel(f"mean error over {reps} runs"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG / "sim_error_vs_T.png", dpi=150); plt.close()
    write_tex("sim3_error_table.tex", tex_table(
        ["$T$"] + [f"{s:,}".replace(",", "{,}") for s in sizes],
        [[r"$\|\hat P-P\|_F$"] + [f"{v:.4f}" for v in err_P], [r"$\|\hat\pi-\pi\|_1$"] + [f"{v:.4f}" for v in err_pi]],
        caption=f"Mean error against the truth over {reps} simulated paths per length (seeds 0--{reps - 1}).",
        label="tab:sim3err"))
    m["simSlopeP"], m["simSlopePi"], m["simReps"], m["simSkipped"] = f"{slope_P:.2f}", f"{slope_pi:.2f}", str(reps), str(skipped)
    ver.append(["Simulation ($T=50$ to $20{,}000$)", r"log--log slope of $\|\hat P-P\|_F$", r"$-1/2$",
                f"slope $= {slope_P:.2f}$", abs(slope_P + 0.5) < 0.1])

    # ------------------------------------------------------------ 3. absorbing chain
    seqs, P_full_true = simulate_absorbing(n_paths=20_000, seed=SEED)
    n_full = my_count_transitions(seqs, K, K + 1)
    P_full_hat = my_row_normalize(n_full)
    Q_hat, R_hat = P_full_hat[:, :K], P_full_hat[:, K:]
    Q_true, R_true = my_canonical_form(P_full_true, [0, 1, 2], [3])
    N_hat, N_true = my_fundamental_matrix(Q_hat), my_fundamental_matrix(Q_true)
    t_hat, t_true = my_absorption_times(N_hat), my_absorption_times(N_true)
    B_hat = my_absorption_probs(N_hat, R_hat)
    starts = np.array([s[0] for s in seqs]); lengths = np.array([len(s) - 1 for s in seqs])
    t_sim = np.array([lengths[starts == i].mean() for i in range(K)])
    t_se = np.array([lengths[starts == i].std(ddof=1) / np.sqrt((starts == i).sum()) for i in range(K)])
    a_N, _ = report("N_hat vs numpy.linalg.inv", N_hat, np.linalg.inv(np.eye(K) - Q_hat))
    print("t true", t_true.round(3), " t from data", t_hat.round(3), " simulated", t_sim.round(3), " B_hat", B_hat.ravel())
    rows = [[str(i), f"{P_OUT[i]:.2f}", f"{t_true[i]:.2f}", f"{t_hat[i]:.2f}", f"{t_sim[i]:.2f} $\\pm$ {1.96 * t_se[i]:.2f}"]
            for i in range(K)]
    write_tex("sim3_absorb.tex", tex_table(
        ["Start", r"$\Pr(\text{absorbed next})$", r"$t$ from true $P$", r"$\hat t=\hat N\mathbf 1$ from counts",
         "Simulated mean (95\\% CI)"], rows,
        caption="Absorbing chain: expected steps before absorption from the truth, from the estimated chain "
        "(our functions) and from 20{,}000 simulated paths.", label="tab:sim3absorb"))
    zt = (np.abs(t_sim - t_true) / t_se).max()
    ver += [["Simulation (absorbing)", r"$\hat N=(I-\hat Q)^{-1}$", r"\texttt{numpy.linalg.inv}", tex_diff(a_N), a_N < 1e-10],
            ["Simulation (absorbing)", r"$t=N\mathbf 1$", "simulated mean time", tex_diff(zt, "z"), zt < 3]]

    # ------------------------------------------------------------ 4. stress test
    xs = simulate_second_order(T=20_000, seed=SEED)
    x2c, dfc = my_markov_test_statistic(x, K)
    x2s, dfs = my_markov_test_statistic(xs, K)
    pc, ps = stats.chi2.sf(x2c, dfc), stats.chi2.sf(x2s, dfs)
    P2s = my_second_order_P(xs, K)
    print(f"Markov test, control chain:  X2 = {x2c:.1f}, df = {dfc}, p = {pc:.3f}")
    print(f"Markov test, second-order chain: X2 = {x2s:.1f}, df = {dfs}, p = {ps:.2e}")
    print("stress-test second-order P [h, i, :]\n", P2s.round(2))
    rows = [[f"$X_t={lab[i]}$" if h == 0 else "", f"$X_{{t-1}}={lab[h]}$"] + [f"{P2s[h, i, j]:.3f}" for j in range(K)]
            for i in range(K) for h in range(K)]
    write_tex("sim3_stress.tex", tex_table(
        ["", "", r"$\Pr(X_{t+1}=0\mid\cdot)$", r"$\Pr(X_{t+1}=1\mid\cdot)$", r"$\Pr(X_{t+1}=2\mid\cdot)$"], rows,
        align="llrrr", caption=r"Stress test: a chain whose next state depends on the last two states "
        r"(after two equal states it repeats with probability 0.8). The rows within a block now differ sharply.",
        label="tab:sim3stress"))
    m["ctrlXtwo"], m["ctrlDf"], m["ctrlP"] = f"{x2c:.1f}", str(dfc), f"{pc:.3f}"
    m["stressXtwo"], m["stressDf"] = f"{x2s:,.0f}".replace(",", "{,}"), str(dfs)
    stress_p = r"< 10^{-300}" if ps == 0 else "= " + tex_num(ps)
    m["stressP"] = stress_p
    ver += [["Simulation (control)", "Markov test", r"$\chi^2_{%d}$ (should not reject)" % dfc, f"$p={pc:.2f}$", pc > 0.05],
            ["Stress test (2nd order)", "Markov test", r"$\chi^2_{%d}$ (should reject)" % dfs, f"$p {stress_p}$", ps < 0.001]]

    # ------------------------------------------------------------ 5. assumptions on simulated data
    print("Assumption checks on simulated data:")
    rows = assumption_checks(x, K, label="")
    rows += assumption_checks(seqs, K, n_full=n_full, label="")[-1:]
    # A6: split the path into 20 "batters" that share the same P; pooling must look harmless
    pieces = np.array_split(x, 20)
    px2, pdf = my_pooling_statistic([my_count_transitions(p, K) for p in pieces])
    pp = stats.chi2.sf(px2, pdf)
    rows.append(["A6 Same $P$ for every batter", rf"20 pseudo-batters: $X^2={px2:.1f}$, df $={pdf}$, $p={pp:.3f}$", pp > 0.05])
    # A7: censor the absorbing paths at an independent geometric time (non-informative by
    #     construction); the censoring rate after a visit must then be the same in every state
    rc = np.random.default_rng(SEED)
    censored_seqs = []
    for s in seqs:
        c_time = rc.geometric(0.04)
        censored_seqs.append(s[:c_time] if c_time < len(s) else s)
    cx2, cdf, rate = my_censoring_statistic(censored_seqs, K)
    cp = stats.chi2.sf(cx2, cdf)
    rows.append(["A7 Non-informative censoring", rf"censoring rate by state ${', '.join(f'{v:.3f}' for v in rate)}$: $X^2={cx2:.1f}$, df $={cdf}$, $p={cp:.3f}$", cp > 0.05])
    rows.append(["A8 ``Not selected'' means dropped", "not applicable: the simulation has no selection decisions", None])
    for r in rows[-3:]:
        print(f"   {r[0]:38s} {r[1]:60s} {'pass' if r[2] else ('n/a' if r[2] is None else 'FAIL')}")
    write_tex("assumptions_sim.tex", tex_table(
        ["Assumption", "Check on the simulated data", "Pass?"],
        [[r[0], r[1], "n/a" if r[2] is None else (r"\checkmark" if r[2] else r"$\times$")] for r in rows], align=r">{\raggedright\arraybackslash}p{0.3\linewidth}>{\raggedright\arraybackslash}p{0.55\linewidth}c",
        caption="Assumption checks on the simulated data, where every assumption holds by construction. "
        "This shows what passing looks like.", label="tab:asssim"))

    m.save()
    log_verification("sim", ver)


if __name__ == "__main__":
    main()
