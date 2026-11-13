# Is Form Real? Modeling Cricket Batting Performance with Markov Chains

DS5020 Intro Linear Algebra & Probability, Fall 2026 (Dr. Fatema Nafa)
Team: Aditya Vishwanath, Adithya Anand, Manavkrishna Manoj

We model IPL batting as a Markov chain at two scales:

* **Chain I, innings level:** states F (0–9 runs), S (10–29), B (30+) and the absorbing state D (not in the team's next match).
* **Chain II, ball level:** states 0, 1, 2, 3, 4, 6 (runs off a ball) and the absorbing state Out.

## Repository layout

```
src/        all code (imports: numpy, pandas, matplotlib, scipy.stats + standard library)
  load_data.py         Iteration 2: load the Cricsheet zip, build both chains, describe the data
  simulate.py          simulated data with known truth; run it for the Iteration 2 simulation check
  methods.py           Iteration 3: from-scratch functions (my_*), one per algorithm step
  small_example.py     Iteration 3: the tiny worked example, step by step, in exact fractions
  check_simulation.py  Iteration 3: recover the truth, compare with numpy, error vs T, stress test
  run_real.py          Iteration 3: real data, library comparison, assumption checks, verification table
  report_utils.py      report() comparison helper and LaTeX writers
data/       raw data (not in Git; see data/README.md)
figures/    figures written by the scripts
docs/       report PDFs, the Overleaf source (docs/report/) and the Excel plan
```

## Setup

1. Download `ipl_json.zip` from <https://cricsheet.org/downloads/> (Indian Premier League, JSON) and put it in `data/`.
   Do not unzip it. The raw file stays out of Git (`.gitignore`).
2. `pip install -r requirements.txt`

## How to run (from the repository root)

| Step | Command | What it does |
|---|---|---|
| 1 | `python src/load_data.py` | Data description and the two data figures (Iteration 2) |
| 2 | `python src/simulate.py` | Iteration 2 simulation check (P-hat, pi, n-step, absorption) |
| 3 | `python src/small_example.py` | Tiny example, every intermediate matrix and row operation printed |
| 4 | `python src/check_simulation.py` | Simulation test: truth, library, error vs T, stress test, assumption checks |
| 5 | `python src/run_real.py` | Real data, library comparison, assumption checks, verification table |

Every number and table in the report is written by these scripts to `docs/report/generated/`, and every figure
to `figures/` (copy it to `docs/report/figures/` before compiling). Run them in the order above, because
`run_real.py` builds the verification table from the outputs of steps 3–5. The random seed is fixed
(`np.random.default_rng(0)`), so every member gets the same numbers. Steps 1–5 take about 15 seconds in total.

## Overleaf

Zip the folder `docs/report/` (it contains `main.tex`, `generated/` and `figures/`) and upload it to Overleaf
with *New Project → Upload Project*. In `main.tex`, `\iterthreefalse` builds the Iteration 2 report
and `\iterthreetrue` builds the report for Iterations 2 and 3.

## Data

Cricsheet ball-by-ball data, <https://cricsheet.org>, released under the Open Data Commons Attribution License (ODC-By).
