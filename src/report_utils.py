"""Helpers shared by all scripts: the course's report() comparison format and small
writers that put script results into docs/report/generated/ so that every number in
the Overleaf report is produced by a script."""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "docs" / "report" / "generated"
FIG = ROOT / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def report(name, mine, ref):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref|| (course format).
    Returns (abs_err, rel_err) so scripts can log them."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    print(f"{name:32s} max|diff| = {abs_err:.2e} rel = {rel_err:.2e}")
    return abs_err, rel_err


# ---------------------------------------------------------------- LaTeX output
def tex_num(v, digits=3):
    """Format a number (int, float or Fraction) for LaTeX."""
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return str(v.numerator)
        sign = "-" if v < 0 else ""
        return rf"{sign}\tfrac{{{abs(v.numerator)}}}{{{v.denominator}}}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", "{,}")
    v = float(v)
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        m, e = f"{v:.1e}".split("e")
        return rf"{m}\times 10^{{{int(e)}}}"
    return f"{v:.{digits}f}"


def tex_matrix(M, digits=3):
    M = np.atleast_2d(np.asarray(M, dtype=object))
    rows = [" & ".join(tex_num(v, digits) for v in r) for r in M]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def tex_table(header, rows, align=None, caption=None, label=None):
    align = align or "l" + "r" * (len(header) - 1)
    out = [r"\begin{table}[H]\centering\small"]
    if caption:
        out.append(rf"\caption{{{caption}}}")
    if label:
        out.append(rf"\label{{{label}}}")
    out += [r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
            rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out)


def write_tex(name, text):
    (GEN / name).write_text(text + "\n")


class Macros:
    """Collect \\newcommand definitions and write them to one file."""

    def __init__(self, filename):
        self.filename, self.items = filename, {}

    def __setitem__(self, key, value):
        assert key.isalpha(), key
        self.items[key] = value

    def save(self):
        wrap = lambda v: rf"\ensuremath{{{v}}}" if any(t in v for t in ("\\times", "\\tfrac", "bmatrix", "^", "<")) else v
        lines = [rf"\newcommand{{\{k}}}{{{wrap(v)}}}" for k, v in self.items.items()]
        write_tex(self.filename, "\n".join(lines))


# ----------------------------------------------------- verification table rows
def log_verification(group, rows):
    """rows: list of (data, quantity, compared_with, difference_tex, passed)."""
    rows = [[str(c) if i < 4 else bool(c) for i, c in enumerate(r)] for r in rows]
    (GEN / f"verification_{group}.json").write_text(json.dumps(rows, indent=1))


def read_verification(groups):
    rows = []
    for g in groups:
        p = GEN / f"verification_{g}.json"
        if p.exists():
            rows += json.loads(p.read_text())
    return rows


def tex_diff(a, kind="max"):
    """'max |.| = 2.2 x 10^-16' style text for the verification table."""
    if a == 0:
        return "0"
    lab = {"max": r"max $|\cdot|", "norm": r"$\|\cdot\|_2", "z": r"max $|\cdot|/\mathrm{se}"}[kind]
    return f"{lab} = {tex_num(a, 2)}$"
