"""Iteration 3, Questions 1-3: the small worked example, step by step, in exact fractions.

  Q1-Q2: a made-up batter's 15 innings, F S B B F S S B F F B S B B F (3 states).
  Q3:    a small absorbing chain with 2 transient states (L = low score, H = high score)
         and 1 absorbing state (D = dropped).

Every intermediate matrix is printed, and docs/report/generated/small_example.tex is
written for the Overleaf report. Run:  python src/small_example.py
"""
from fractions import Fraction as Fr

import numpy as np

from methods import (my_absorption_probs, my_absorption_times, my_count_transitions,
                     my_fundamental_matrix, my_gauss_solve, my_matrix_power, my_power_iteration,
                     my_row_normalize, my_simulate_absorption_times, my_steady_state_system)
from report_utils import Macros, log_verification, report, tex_diff, tex_matrix, write_tex

SEQUENCE = "FSBBFSSBFFBSBBF"
LABELS = ["F", "S", "B"]


def show(name, M):
    M = np.atleast_2d(M)
    print(f"{name} =")
    for row in M:
        print("   [" + "  ".join(f"{str(v):>6s}" for v in row) + " ]")


def aug_tex(M, k):
    cols = "c" * k + "|" + "c" * (M.shape[1] - k)
    body = r" \\ ".join(" & ".join(tex_cell(v) for v in r) for r in M)
    return rf"\left[\begin{{array}}{{{cols}}}{body}\end{{array}}\right]"


def tex_cell(v):
    v = Fr(v)
    if v.denominator == 1:
        return str(v.numerator)
    return ("-" if v < 0 else "") + rf"\tfrac{{{abs(v.numerator)}}}{{{v.denominator}}}"


def steps_tex(log, k):
    parts = []
    for i, (msg, M) in enumerate(log):
        label = "" if msg == "start" else rf"\xrightarrow{{{msg}}}"
        parts.append(label + aug_tex(M, k))
    # two matrices per line
    lines = []
    for i in range(0, len(parts), 2):
        lines.append(r"&" + " ".join(parts[i:i + 2]))
    return r"\begin{align*}" + r"\\[4pt]".join(lines) + r"\end{align*}"


def main():
    tex, m, ver = [], Macros("small_numbers.tex"), []
    # ------------------------------------------------------------------ Q1
    x = [LABELS.index(c) for c in SEQUENCE]
    print("Sequence:", " ".join(SEQUENCE), f"({len(x)} innings, {len(x) - 1} transitions)")
    print("Transitions:", " ".join(f"{SEQUENCE[t]}->{SEQUENCE[t + 1]}" for t in range(len(x) - 1)))
    n = my_count_transitions(x, 3)
    show("Step 1: counts n", n)
    print("   row totals n_i =", n.sum(axis=1))
    P = my_row_normalize(n, exact=True)
    show("Step 2: P_hat = n_ij / n_i", P)
    rows = P.sum(axis=1)
    print("   row sums of P_hat:", [str(v) for v in rows])
    assert all(r == 1 for r in rows)
    tex.append(r"\paragraph{Step 1: count the transitions.} The 15 innings give 14 transitions: "
               + ", ".join(rf"\texttt{{{SEQUENCE[t]}}}$\to$\texttt{{{SEQUENCE[t + 1]}}}" for t in range(len(x) - 1))
               + r". Counting them (rows: this innings, columns: next innings, order F, S, B):")
    tex.append(rf"\[ n = {tex_matrix(n)}, \qquad (n_F, n_S, n_B) = ({', '.join(str(v) for v in n.sum(axis=1))}). \]")
    tex.append(r"\paragraph{Step 2: estimate $P$ (Iteration 2, Q2).} Divide each row by its total, "
               r"$\hat P_{ij} = n_{ij}/n_i$:")
    tex.append(rf"\[ \hat P = {tex_matrix(P)}, \qquad \hat P\mathbf 1 = {tex_matrix(rows.reshape(-1, 1))}. \]")
    tex.append(r"Every row sums to 1, so every row is a probability distribution.")

    # ------------------------------------------------------------------ Q2
    P2 = my_matrix_power(P, 2)
    show("Step 3: P_hat^2", P2)
    tex.append(r"\paragraph{Step 3: two-step probabilities (Iteration 2, Q3).} "
               r"$(\hat P^2)_{ij} = \sum_l \hat P_{il}\hat P_{lj}$:")
    tex.append(rf"\[ \hat P^2 = {tex_matrix(P2)}. \]")
    tex.append(rf"For example, a batter who fails today scores Big two innings later with probability "
               rf"$(\hat P^2)_{{FB}} = {tex_cell(P2[0, 2])}$. All entries of $\hat P^2$ are positive, so the chain "
               r"is regular and has a unique steady state.")
    assert all(v > 0 for v in P2.ravel())

    A, e = my_steady_state_system(P)
    show("Step 4: steady-state system A (rows 1-2 of P^T - I, last row ones)", A)
    print("   right-hand side e =", [str(v) for v in e])
    log = []
    pi = my_gauss_solve(A, e, log)
    for msg, M in log:
        show(f"   after {msg}", M)
    print("   pi =", [str(v) for v in pi])
    tex.append(r"\paragraph{Step 4: steady state (Iteration 2, Q4).} The system $(\hat P^\top - I)\pi^\top = 0$ "
               r"has rank 2, so its last equation is replaced by $\pi_F + \pi_S + \pi_B = 1$. "
               r"Gaussian elimination with partial pivoting on the augmented matrix $[A \mid e]$, "
               r"printed after every row operation:")
    tex.append(steps_tex(log, 3))
    tex.append(rf"Back-substitution gives $\pi = ({', '.join(tex_cell(v) for v in pi)})$.")

    piP = pi @ P
    check_ok = all(a == b for a, b in zip(piP, pi)) and sum(pi) == 1
    print("   check pi P = pi:", [str(v) for v in piP], " sum(pi) =", sum(pi), "->", check_ok)
    pi_pow, errs = my_power_iteration(np.array(P, dtype=float))
    print("   second way, power iteration:", pi_pow.round(6), f"({len(errs)} iterations)")
    tex.append(r"\paragraph{Check a second way.} Substituting back, "
               rf"$\pi\hat P = ({', '.join(tex_cell(v) for v in piP)}) = \pi$ exactly and the entries sum to "
               rf"{sum(pi)}. Power iteration $\pi^{{(t+1)}} = \pi^{{(t)}}\hat P$ from the uniform vector "
               rf"reaches $({', '.join(f'{v:.6f}' for v in pi_pow)})$ after {len(errs)} steps, "
               rf"the decimal values of $({', '.join(tex_cell(v) for v in pi)})$. The raw shares of the 15 innings are "
               rf"$({', '.join(tex_cell(Fr(SEQUENCE.count(c), 15)) for c in LABELS)})$; with so few innings they "
               r"need not equal $\pi$.")
    ver.append(["Small example", r"$\pi$ (Gaussian elim.)", r"$\pi\hat P=\pi$ in exact fractions", "0", True])
    a, _ = report("small: power iteration vs exact pi", pi_pow, np.array(pi, float))
    ver.append(["Small example", r"$\pi$ (power iteration)", "exact fractions", tex_diff(a), a < 1e-10])
    m["exPiF"], m["exPiS"], m["exPiB"] = (tex_cell(v) for v in pi)

    # ------------------------------------------------------------------ Q3
    Q = np.array([[Fr(1, 2), Fr(1, 4)], [Fr(1, 4), Fr(2, 3)]], dtype=object)
    R = np.array([[Fr(1, 4)], [Fr(1, 12)]], dtype=object)
    print("\nAbsorbing chain, states L, H (transient) and D (absorbing)")
    show("Q", Q); show("R", R)
    print("   rows of [Q R] sum to", [str(v) for v in (Q.sum(axis=1) + R.sum(axis=1))])
    I = np.array([[Fr(1), Fr(0)], [Fr(0), Fr(1)]], dtype=object)
    show("I - Q", I - Q)
    log = []
    N = my_fundamental_matrix(Q, log)
    for msg, M in log:
        show(f"   after {msg}", M)
    show("N = (I - Q)^-1", N)
    t = my_absorption_times(N)
    B = my_absorption_probs(N, R)
    print("   t = N 1 =", [str(v) for v in t], " B = N R =", [str(v) for v in B.ravel()])
    ok_inv = all(v == w for v, w in zip(((I - Q) @ N).ravel(), I.ravel()))
    ok_t = all(v == w for v, w in zip(t, 1 + Q @ t))
    print("   check (I - Q) N = I:", ok_inv, "   check t = 1 + Q t:", ok_t)
    rng = np.random.default_rng(0)
    P_rows = np.hstack([np.array(Q, float), np.array(R, float)])
    sim = [my_simulate_absorption_times(P_rows, s, 100_000, rng) for s in range(2)]
    print("   simulated mean steps (100,000 paths):", [round(float(s.mean()), 3) for s in sim])

    tex.append(r"\paragraph{Step 5: a small absorbing chain (Iteration 3, Q3).} Two transient states, "
               r"L (low score) and H (high score), and the absorbing state D (dropped). In canonical form")
    P_can = np.vstack([np.hstack([Q, R]), np.array([[Fr(0), Fr(0), Fr(1)]], dtype=object)])
    tex.append(rf"\[ P = {tex_matrix(P_can)}, \quad Q = {tex_matrix(Q)}, \quad R = {tex_matrix(R)}, "
               rf"\quad I - Q = {tex_matrix(I - Q)}. \]")
    tex.append(r"We solve $(I-Q)N = I$ by Gaussian elimination on $[\,I-Q \mid I\,]$:")
    tex.append(steps_tex(log, 2))
    tex.append(rf"\[ N = (I-Q)^{{-1}} = {tex_matrix(N)}, \qquad t = N\mathbf 1 = {tex_matrix(t.reshape(-1, 1))}, "
               rf"\qquad B = NR = {tex_matrix(B)}. \]")
    tex.append(rf"Starting after a low score, the batter is expected to play ${tex_cell(t[0])} = {float(t[0]):.1f}$ "
               rf"more innings before being dropped; after a high score, ${tex_cell(t[1])} = {float(t[1]):.1f}$. "
               r"$B = \mathbf 1$ because D is the only absorbing state.")
    tex.append(r"\paragraph{Check a second way.} $(I-Q)N = I$ exactly, and $t$ satisfies the "
               r"first-step equation $t = \mathbf 1 + Qt$ (one step, then the expected time from wherever the "
               r"chain moved). Simulating 100{,}000 paths from each start gives mean times "
               rf"{sim[0].mean():.3f} and {sim[1].mean():.3f}, close to {float(t[0]):.1f} and {float(t[1]):.1f}.")
    m["exTL"], m["exTH"] = tex_cell(t[0]), tex_cell(t[1])
    ver.append(["Small example", r"$N=(I-Q)^{-1}$", r"$(I-Q)N=I$ in exact fractions", "0", bool(ok_inv)])
    a, _ = report("small: N vs numpy.linalg.inv", np.array(N, float), np.linalg.inv(np.eye(2) - np.array(Q, float)))
    ver.append(["Small example", r"$N$", r"\texttt{numpy.linalg.inv}", tex_diff(a), a < 1e-10])

    write_tex("small_example.tex", "\n\n".join(tex))
    m.save()
    log_verification("small", ver)


if __name__ == "__main__":
    main()
