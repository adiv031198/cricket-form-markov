"""From-scratch implementation of every step of the algorithm in the report
(Section "The Algorithm as Math"). Only NumPy arrays and matrix products are used;
numpy.linalg is never called here. Every function works on float arrays and on
object arrays of fractions.Fraction, so the same code runs the small example
(exact), the simulation and the real data.

State encoding: a chain with k states uses the integers 0, ..., k-1. A sequence is a
1-D integer array or list; a list of sequences is used when the data come in
separate pieces (one innings, or one spell of selection), so that no transition is
counted from the end of one piece to the start of the next.
"""
from fractions import Fraction

import numpy as np


def _as_list_of_sequences(x):
    if len(x) > 0 and np.ndim(x[0]) == 0:
        return [np.asarray(x, dtype=int)]
    return [np.asarray(s, dtype=int) for s in x]


def _identity(n, exact=False):
    if exact:
        return np.array([[Fraction(int(i == j)) for j in range(n)] for i in range(n)], dtype=object)
    return np.eye(n)


def _is_exact(M):
    return np.asarray(M).dtype == object


# Step 1 (counting):  n_ij = #{t : X_t = i, X_{t+1} = j},  n_i = sum_j n_ij
#   Transitions out of a state >= k_from (absorbing) or into a state >= k_to are not counted.
def my_count_transitions(x, k_from, k_to=None):
    k_to = k_to or k_from
    n = np.zeros((k_from, k_to), dtype=int)
    for s in _as_list_of_sequences(x):
        a, b = s[:-1], s[1:]
        keep = (a < k_from) & (b < k_to)        # rows: transient states only; columns: states kept
        np.add.at(n, (a[keep], b[keep]), 1)
    return n


# Step 2 (MLE, Iteration 2 Q2):  P_hat_ij = n_ij / n_i,  i.e.  P_hat = D^{-1} n with D = diag(n_i)
def my_row_normalize(n, exact=False):
    n = np.asarray(n)
    n_i = n.sum(axis=1)
    if (n_i == 0).any():
        raise ValueError("a state with no observed transitions has no estimated row")
    if exact:
        return np.array([[Fraction(int(c), int(r)) for c in row] for row, r in zip(n, n_i)], dtype=object)
    return n / n_i[:, None]


def my_estimate_P(x, k, exact=False):
    """Steps 1-2 together. Returns (P_hat, n_i)."""
    n = my_count_transitions(x, k)
    return my_row_normalize(n, exact), n.sum(axis=1)


# Standard error of a proportion:  se(P_hat_ij) = sqrt(P_hat_ij (1 - P_hat_ij) / n_i)
def my_standard_errors(P_hat, n_i):
    P_hat = np.asarray(P_hat, float)
    return np.sqrt(P_hat * (1 - P_hat) / np.asarray(n_i)[:, None])


# Matrix product:  (AB)_ij = sum_l A_il B_lj
def my_matmul(A, B):
    return A @ B


# Step 3 (Chapman-Kolmogorov, Iteration 2 Q3):  P^(n) = P^n, by repeated squaring
#   P^n = (P^{n/2})^2 if n even,  P (P^{n-1}) if n odd
def my_matrix_power(P, n):
    result = _identity(P.shape[0], _is_exact(P))
    base = P.copy()
    while n > 0:
        if n & 1:
            result = my_matmul(result, base)
        base = my_matmul(base, base)
        n >>= 1
    return result


# Gaussian elimination with partial pivoting for A x = b (b may have several columns):
#   for each column c: swap the row with the largest |A_rc| into row c, then
#   R_r <- R_r - (A_rc / A_cc) R_c for r > c; finally back-substitute
#   x_r = (b_r - sum_{l>r} A_rl x_l) / A_rr.
def my_gauss_solve(A, b, log=None):
    """If log is a list, a (description, augmented matrix) pair is appended after every
    row operation, so the small example can print each step."""
    exact = _is_exact(A)
    A = np.array(A, dtype=object if exact else float)
    b = np.array(b, dtype=object if exact else float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    n = A.shape[0]
    snap = lambda msg: log.append((msg, np.hstack([A, b]).copy())) if log is not None else None
    snap("start")
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(A[r, c]))
        if A[piv, c] == 0:
            raise ValueError("matrix is singular")
        if piv != c:
            A[[c, piv]], b[[c, piv]] = A[[piv, c]], b[[piv, c]]
            snap(rf"R_{{{c + 1}}} \leftrightarrow R_{{{piv + 1}}}")
        for r in range(c + 1, n):
            f = A[r, c] / A[c, c]
            if f != 0:
                A[r, c:] = A[r, c:] - f * A[c, c:]
                b[r] = b[r] - f * b[c]
                snap(rf"R_{{{r + 1}}} \leftarrow R_{{{r + 1}}} - ({_fmt(f)})\,R_{{{c + 1}}}")
    x = np.empty_like(b)
    for r in range(n - 1, -1, -1):
        s = b[r] - (A[r, r + 1:] @ x[r + 1:] if r + 1 < n else 0)
        x[r] = s / A[r, r]
    return x[:, 0] if x.shape[1] == 1 else x


def _fmt(f):
    if isinstance(f, Fraction):
        return str(f) if f.denominator != 1 else str(f.numerator)
    return f"{float(f):.3g}"


# Step 4 (steady state, Iteration 2 Q4):  pi P = pi, sum_i pi_i = 1.
#   Write as (P^T - I) pi^T = 0, replace its last (redundant) equation by 1^T pi^T = 1:
#   A pi^T = e_k  with  A = [rows 1..k-1 of (P^T - I); 1^T].
def my_steady_state_system(P):
    k = P.shape[0]
    exact = _is_exact(P)
    A = P.T - _identity(k, exact)
    A[-1, :] = Fraction(1) if exact else 1.0
    e = np.array([Fraction(0)] * (k - 1) + [Fraction(1)], dtype=object) if exact else np.r_[np.zeros(k - 1), 1.0]
    return A, e


def my_steady_state(P, log=None):
    A, e = my_steady_state_system(P)
    return my_gauss_solve(A, e, log)


# Power iteration for pi (second method):  pi^(t+1) = pi^(t) P,  starting from the uniform vector
def my_power_iteration(P, tol=1e-13, max_iter=10_000):
    pi = np.full(P.shape[0], 1.0 / P.shape[0])
    errors = []
    for _ in range(max_iter):
        new = pi @ P
        errors.append(np.abs(new - pi).sum())
        pi = new
        if errors[-1] < tol:
            break
    return pi, np.array(errors)


# Step 5a (canonical form):  P = [[Q, R], [0, I]] with Q = P[T, T], R = P[T, A]
def my_canonical_form(P_full, transient, absorbing):
    P_full = np.asarray(P_full)
    return P_full[np.ix_(transient, transient)], P_full[np.ix_(transient, absorbing)]


# Step 5b (fundamental matrix, Iteration 2 Q5):  solve (I - Q) N = I,  so N = (I - Q)^{-1} = sum_k Q^k
def my_fundamental_matrix(Q, log=None):
    I = _identity(Q.shape[0], _is_exact(Q))
    return my_gauss_solve(I - Q, I, log)


# Step 5c:  expected number of steps before absorption  t = N 1
def my_absorption_times(N):
    one = np.array([Fraction(1)] * N.shape[0], dtype=object) if _is_exact(N) else np.ones(N.shape[0])
    return N @ one


# Step 5d:  absorption probabilities  B = N R
def my_absorption_probs(N, R):
    return N @ R


# Variance of the absorption time:  Var(T) = (2N - I) t - t o t   (o = elementwise product)
def my_absorption_time_variance(N, t):
    return (2 * N - _identity(N.shape[0], _is_exact(N))) @ t - t * t


# Markov-property test (Iteration 3 Q5):
#   P2[h, i, j] = Pr(X_{t+1} = j | X_t = i, X_{t-1} = h) = n_hij / n_hi
def my_second_order_counts(x, k):
    c = np.zeros((k, k, k), dtype=int)
    for s in _as_list_of_sequences(x):
        if len(s) < 3:
            continue
        h, i, j = s[:-2], s[1:-1], s[2:]
        keep = (h < k) & (i < k) & (j < k)
        np.add.at(c, (h[keep], i[keep], j[keep]), 1)
    return c


def my_second_order_P(x, k):
    c = my_second_order_counts(x, k).astype(float)
    tot = c.sum(axis=2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, c / tot, np.nan)


#   For each current state i, a chi-squared test of homogeneity of the k x k table
#   (rows: previous state h, columns: next state j):
#   E_hj = n_h. n_.j / n_..,  X_i^2 = sum (O - E)^2 / E,  df_i = (rows - 1)(cols - 1).
#   Under the Markov property X^2 = sum_i X_i^2 is approximately chi^2 with df = sum_i df_i.
def my_markov_test_statistic(x, k):
    c = my_second_order_counts(x, k).astype(float)
    x2, df = 0.0, 0
    for i in range(k):
        T = c[:, i, :]
        T = T[T.sum(axis=1) > 0][:, T.sum(axis=0) > 0]
        if T.shape[0] < 2 or T.shape[1] < 2:
            continue
        E = T.sum(axis=1, keepdims=True) * T.sum(axis=0, keepdims=True) / T.sum()
        x2 += ((T - E) ** 2 / E).sum()
        df += (T.shape[0] - 1) * (T.shape[1] - 1)
    return x2, df


#   Same statistic for "are two transition-count matrices drawn from the same P?"
#   (used for the time-homogeneity check): for each row i, a 2 x k homogeneity table.
def my_homogeneity_statistic(n_a, n_b):
    x2, df = 0.0, 0
    for i in range(n_a.shape[0]):
        T = np.vstack([n_a[i], n_b[i]]).astype(float)
        T = T[:, T.sum(axis=0) > 0]
        if (T.sum(axis=1) == 0).any() or T.shape[1] < 2:
            continue
        E = T.sum(axis=1, keepdims=True) * T.sum(axis=0, keepdims=True) / T.sum()
        x2 += ((T - E) ** 2 / E).sum()
        df += T.shape[1] - 1
    return x2, df


#   A6 (same P for every batter): for each row i, a homogeneity table with one row per batter
#   (batters with at least min_row transitions out of i) and one column per next state.
def my_pooling_statistic(count_matrices, min_row=30):
    x2, df = 0.0, 0
    for i in range(count_matrices[0].shape[0]):
        T = np.array([c[i] for c in count_matrices], dtype=float)
        T = T[T.sum(axis=1) >= min_row]
        T = T[:, T.sum(axis=0) > 0]
        if T.shape[0] < 2 or T.shape[1] < 2:
            continue
        E = T.sum(axis=1, keepdims=True) * T.sum(axis=0, keepdims=True) / T.sum()
        x2 += ((T - E) ** 2 / E).sum()
        df += (T.shape[0] - 1) * (T.shape[1] - 1)
    return x2, df


#   A7 (non-informative censoring): the chance that a sequence is censored right after a visit
#   to state i must not depend on i. Table (k x 2): visits to i followed by censoring vs not;
#   X^2 homogeneity statistic with df = k - 1.
def my_censoring_statistic(seqs, k):
    visits = np.zeros(k); cens = np.zeros(k)
    for s in _as_list_of_sequences(seqs):
        tr = s[s < k]
        visits += np.bincount(tr, minlength=k)[:k]
        if len(s) and s[-1] < k:                    # ended without absorption: censored
            cens[s[-1]] += 1
    T = np.column_stack([cens, visits - cens])
    E = T.sum(axis=1, keepdims=True) * T.sum(axis=0, keepdims=True) / T.sum()
    return ((T - E) ** 2 / E).sum(), k - 1, cens / visits


# Simulation of absorption times from a full transition matrix (rows: transient states)
#   T = number of steps until the first visit to an absorbing state
def my_simulate_absorption_times(P_trans_rows, start, n_paths, rng):
    P_trans_rows = np.asarray(P_trans_rows, float)
    m = P_trans_rows.shape[0]
    cum = np.cumsum(P_trans_rows, axis=1)
    cum[:, -1] = 1.0
    state = np.full(n_paths, start)
    steps = np.zeros(n_paths, dtype=int)
    alive = np.ones(n_paths, dtype=bool)
    while alive.any():
        steps[alive] += 1
        u = rng.random(alive.sum())
        state[alive] = (u[:, None] > cum[state[alive]]).sum(axis=1)
        alive &= state < m
    return steps
