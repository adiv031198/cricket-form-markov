"""Simulated data with a known answer (Iteration 2 model, reused unchanged in Iteration 3).

Model: a 3-state Markov chain with the transition matrix P_TRUE from the course
handout, seed 0. For the absorbing-chain question (Q5) every state has its own
probability of moving to an absorbing state "Out"/"Dropped" (P_OUT), so that the
expected absorption times differ between states.

Run  `python src/simulate.py`  for the Iteration 2 simulation check.
"""
import numpy as np

P_TRUE = np.array([[0.6, 0.3, 0.1],
                   [0.2, 0.5, 0.3],
                   [0.1, 0.4, 0.5]])
P_OUT = np.array([0.15, 0.05, 0.02])   # Pr(absorbed next step | current state), states 0, 1, 2
SEED = 0


def simulate_chain(P=P_TRUE, T=20_000, seed=SEED):
    """One path X_0, ..., X_{T-1} with X_0 = 0 (exactly the handout's code).
    Returns (x, P_true)."""
    rng = np.random.default_rng(seed)
    k = P.shape[0]
    x = np.zeros(T, dtype=int)
    for t in range(1, T):
        x[t] = rng.choice(k, p=P[x[t - 1]])
    return x, P


def absorbing_matrix(P=P_TRUE, p_out=P_OUT):
    """Full (k+1) x (k+1) matrix in canonical order: transient states 0..k-1, absorbing state k.
    Row i: (1 - p_out_i) P_i for the transient states and p_out_i for absorption."""
    k = P.shape[0]
    P_full = np.zeros((k + 1, k + 1))
    P_full[:k, :k] = (1 - p_out)[:, None] * P
    P_full[:k, k] = p_out
    P_full[k, k] = 1.0
    return P_full


def simulate_absorbing(n_paths=20_000, P=P_TRUE, p_out=P_OUT, seed=SEED):
    """n_paths sequences that each start in a uniformly chosen transient state and stop
    at the first visit to the absorbing state k. Returns (sequences, P_full)."""
    rng = np.random.default_rng(seed)
    P_full = absorbing_matrix(P, p_out)
    k = P.shape[0]
    cum = np.cumsum(P_full, axis=1)
    cum[:, -1] = 1.0
    seqs = []
    for _ in range(n_paths):
        s = [rng.integers(k)]
        while s[-1] != k:
            s.append(int((rng.random() > cum[s[-1]]).sum()))
        seqs.append(np.array(s))
    return seqs, P_full


def simulate_n_step_fractions(P=P_TRUE, n=3, n_paths=20_000, seed=SEED):
    """F[i, j] = fraction of n_paths simulated paths started at i that are at j after n steps
    (estimates (P^n)_ij)."""
    rng = np.random.default_rng(seed)
    k = P.shape[0]
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0
    F = np.zeros((k, k))
    for i in range(k):
        state = np.full(n_paths, i)
        for _ in range(n):
            state = (rng.random(n_paths)[:, None] > cum[state]).sum(axis=1)
        F[i] = np.bincount(state, minlength=k) / n_paths
    return F


def simulate_second_order(T=20_000, stick=0.8, P=P_TRUE, seed=SEED):
    """Stress test: a chain that is NOT Markov. After two equal states in a row the next state
    repeats with probability `stick`; otherwise the next state is drawn from P[X_t].
    Returns x."""
    rng = np.random.default_rng(seed)
    k = P.shape[0]
    x = np.zeros(T, dtype=int)
    for t in range(1, T):
        if t >= 2 and x[t - 1] == x[t - 2]:
            others = [j for j in range(k) if j != x[t - 1]]
            x[t] = x[t - 1] if rng.random() < stick else rng.choice(others)
        else:
            x[t] = rng.choice(k, p=P[x[t - 1]])
    return x


# ----------------------------------------------------------------------------
# Iteration 2 simulation check (library routines are allowed here: Python is only
# used to test the derivations numerically before the from-scratch code exists).
# ----------------------------------------------------------------------------
def iteration2_check():
    from report_utils import Macros, tex_num, tex_table, write_tex

    x, P = simulate_chain(T=20_000, seed=SEED)
    k, T = 3, len(x)
    n = np.zeros((k, k))
    np.add.at(n, (x[:-1], x[1:]), 1)
    P_hat = n / n.sum(axis=1, keepdims=True)                     # Q2
    se = np.sqrt(P * (1 - P) / n.sum(axis=1)[:, None])

    A = np.vstack([P.T - np.eye(k), np.ones(k)])                 # Q4: add the sum-to-one row
    b = np.r_[np.zeros(k), 1]
    pi_true = np.linalg.lstsq(A, b, rcond=None)[0]
    pi_hat = np.linalg.lstsq(np.vstack([P_hat.T - np.eye(k), np.ones(k)]), b, rcond=None)[0]
    freq = np.bincount(x, minlength=k) / T
    rank_alone = np.linalg.matrix_rank(P.T - np.eye(k))

    nstep = 3                                                    # Q3
    Pn = np.linalg.matrix_power(P, nstep)
    F = simulate_n_step_fractions(P, n=nstep, n_paths=20_000, seed=SEED)

    seqs, P_full = simulate_absorbing(n_paths=20_000, seed=SEED)  # Q5
    Q, R = P_full[:k, :k], P_full[:k, k:]
    N = np.linalg.inv(np.eye(k) - Q)
    t_true, B = N @ np.ones(k), N @ R
    starts = np.array([s[0] for s in seqs])
    lengths = np.array([len(s) - 1 for s in seqs])
    t_sim = np.array([lengths[starts == i].mean() for i in range(k)])
    t_se = np.array([lengths[starts == i].std(ddof=1) / np.sqrt((starts == i).sum()) for i in range(k)])

    print("P_hat =\n", P_hat.round(3), "\nrow sums:", P_hat.sum(axis=1))
    print("max |P_hat - P| / se =", (np.abs(P_hat - P) / se).max().round(2))
    print("pi true", pi_true.round(4), " pi from P_hat", pi_hat.round(4), " long-run freq", freq.round(4))
    print("rank of (P^T - I) on its own:", rank_alone)
    print(f"P^{nstep} =\n", Pn.round(4), "\nsimulated fractions =\n", F.round(4))
    print("t = N1 =", t_true.round(3), " simulated mean steps =", t_sim.round(3), " B = NR =", B.ravel())

    lab = ["0", "1", "2"]
    rows = []
    for i in range(k):
        for j in range(k):
            rows.append([f"$P_{{{lab[i]}{lab[j]}}}$", tex_num(P[i, j], 2), tex_num(P_hat[i, j]),
                         tex_num(abs(P_hat[i, j] - P[i, j]) / se[i, j], 2)])
    write_tex("it2_sim_P.tex", tex_table(
        ["Entry", "True", r"Estimate $\hat P_{ij}$", r"$|\hat P_{ij}-P_{ij}|/\mathrm{se}$"], rows,
        caption=rf"Simulation check, Q2: transition matrix estimated from one path of length $T={T:,}$.".replace(",", "{,}"),
        label="tab:it2simP"))
    rows = [[f"$\\pi_{i}$", tex_num(pi_true[i], 4), tex_num(pi_hat[i], 4), tex_num(freq[i], 4)] for i in range(k)]
    write_tex("it2_sim_pi.tex", tex_table(
        ["", r"True $\pi$ (from $P$)", r"From $\hat P$", "Long-run frequency"], rows,
        caption=r"Simulation check, Q4: steady state.", label="tab:it2simpi"))
    rows = [[f"$({lab[i]},{lab[j]})$", tex_num(Pn[i, j], 4), tex_num(F[i, j], 4)] for i in range(k) for j in range(k)]
    write_tex("it2_sim_nstep.tex", tex_table(
        [r"$(i,j)$", rf"$(P^{nstep})_{{ij}}$", "Fraction of 20{,}000 simulated paths"], rows,
        caption=rf"Simulation check, Q3: {nstep}-step transition probabilities.", label="tab:it2simnstep"))
    rows = [[f"{i}", tex_num(P_OUT[i], 2), tex_num(t_true[i], 2), f"{t_sim[i]:.2f} $\\pm$ {1.96 * t_se[i]:.2f}",
             tex_num(B[i, 0], 4)] for i in range(k)]
    write_tex("it2_sim_absorb.tex", tex_table(
        ["Start state", r"$\Pr(\text{absorbed next})$", r"$t=N\mathbf 1$", "Simulated mean (95\\% CI)", r"$B=NR$"], rows,
        caption=r"Simulation check, Q5: expected number of steps before absorption (20{,}000 simulated paths).",
        label="tab:it2simabsorb"))

    m = Macros("it2_sim_numbers.tex")
    m["simT"] = f"{T:,}".replace(",", "{,}")
    m["simMaxZ"] = f"{(np.abs(P_hat - P) / se).max():.2f}"
    m["simRowSumErr"] = tex_num(np.abs(P_hat.sum(axis=1) - 1).max())
    m["simRankAlone"] = str(rank_alone)
    m["simPiErr"] = tex_num(np.abs(pi_hat - freq).max())
    m["simNstepErr"] = tex_num(np.abs(Pn - F).max())
    m["simTErr"] = f"{np.abs(t_sim - t_true).max():.2f}"
    m.save()


if __name__ == "__main__":
    iteration2_check()
