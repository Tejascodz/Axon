"""Draw the README's results figure from the committed result files.

    python scripts/make_figures.py            # writes docs/figures/results.svg and results.png

Every value is read from results/; nothing is typed in by hand. Error bars are 95% Wilson intervals
for proportions. The BRIGHT value is a mean nDCG@10 over three domains; per-query scores were not
kept, so it has no interval.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
BASE, AXON, INK, MUTED, GRID = "#b8bec7", "#2457c5", "#1f2328", "#59636e", "#e6e8eb"


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def wilson(p, n, z=1.96):
    """95% Wilson score interval for a proportion p observed over n trials."""
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def values():
    """(label, n, baseline, axon, is_proportion) for each panel, read from results/."""
    r8 = load("results/r8_run1/r8.json")
    r10 = load("results/r10_run1/r10.json")
    r10b = load("results/r10b_run1/r10b.json")
    hp = r8["hotpot_second_hop"]
    mq = r10["musique_full"]
    br = r10["bright_tuned"]["test"]
    bright_n = sum(v["chosen_with_expansion"]["queries"] for v in br.values())
    mean = lambda key: sum(v[key]["final_ndcg@10"] for v in br.values()) / len(br)
    retrieval = [
        ("HotpotQA\nboth gold passages in top 5", hp["test_questions"],
         hp["configs"]["one_step"]["test"]["both_gold@5"], hp["configs"]["titles_slots2"]["test"]["both_gold@5"], True),
        ("MuSiQue\nall supporting passages in top 10", mq["questions"],
         mq["one_step"]["all_supporting@10"], mq["second_hop_follow_ups"]["all_supporting@10"], True),
        ("BRIGHT nDCG@10\nmean of 3 held-out domains", bright_n, mean("round8_setting"), mean("chosen_with_expansion"), False),
    ]
    inj = r10b["injection2"]["end_to_end"]
    n_attacks = 2 * r10b["injection2"]["test_questions"]          # each question attacked at the end and the middle
    pois = r10["poisoning2"]
    attacks = [
        ("Injections written by an LLM\nduring the run (unseen)", n_attacks,
         inj["patterns"]["attack_success_generated"], inj["patterns_and_classifier"]["attack_success_generated"], True),
        ("Hand-written injections", n_attacks,
         inj["patterns"]["attack_success_handwritten"], inj["patterns_and_classifier"]["attack_success_handwritten"], True),
        ("Planted documents restating\nthe question (false answers)", pois["questions"],
         pois["echo"]["none"]["false_answer_given"], pois["echo"]["echo_filter"]["false_answer_given"], True),
    ]
    return retrieval, attacks


def panel(ax, rows, title, subtitle, pct):
    y = list(range(len(rows)))[::-1]
    h = 0.36
    def fmt(v):
        if not pct:
            return f"{v:.3f}"
        if v == 0:
            return "0%"
        return f"{v * 100:.2f}".rstrip("0").rstrip(".") + "%" if v < 0.01 else f"{v * 100:.1f}%"

    def upper(val, n, prop):
        return wilson(val, n)[1] if prop else val
    top = max(max(upper(r[2], r[1], r[4]), upper(r[3], r[1], r[4])) for r in rows)
    for yi, (label, n, base, axon, prop) in zip(y, rows):
        for off, val, color in ((h / 2, base, BASE), (-h / 2, axon, AXON)):
            err = None
            if prop:
                lo, hi = wilson(val, n)
                err = [[val - lo], [hi - val]]
            ax.barh(yi + off, val, height=h, color=color, xerr=err, ecolor=MUTED,
                    error_kw={"elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9})
            ax.text(upper(val, n, prop) + top * 0.025, yi + off, fmt(val),
                    va="center", ha="left", fontsize=9.5, color=INK if color == AXON else MUTED,
                    fontweight="bold" if color == AXON else "normal")
        ax.text(-top * 0.03, yi, label, va="center", ha="right", fontsize=9.5, color=INK, linespacing=1.25)
        ax.text(-top * 0.03, yi - 0.47, f"n = {n:,}", va="center", ha="right", fontsize=8, color=MUTED)
    ax.set_xlim(0, top * 1.25)
    ax.set_ylim(-0.75, len(rows) - 0.35)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0)
    if pct:
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED, ha="left", va="bottom")


def main():
    retrieval, attacks = values()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path", "svg.hashsalt": "axon", "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.3), gridspec_kw={"wspace": 0.95})
    panel(a, retrieval, "Retrieval quality", "Higher is better", pct=False)
    panel(b, attacks, "Attack success", "Lower is better", pct=True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BASE), plt.Rectangle((0, 0), 1, 1, color=AXON)]
    fig.legend(handles, ["One search, pattern scanner only (baseline)", "Axon pipeline (second hop, follow-up searches, "
                         "query expansion, injection classifier, echo filter)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.2, right=0.985, top=0.84, bottom=0.17)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "results.svg"), metadata={"Date": None, "Creator": None})
    fig.savefig(os.path.join(OUT, "results.png"), dpi=200)
    for name, rows in (("retrieval", retrieval), ("attacks", attacks)):
        for label, n, base, axon, prop in rows:
            ci = lambda v: f" [{wilson(v, n)[0]:.4f}, {wilson(v, n)[1]:.4f}]" if prop else ""
            print(f"{name:9s} {label.replace(chr(10), ' '):60s} n={n:5d}  {base:.4f}{ci(base)} -> {axon:.4f}{ci(axon)}")


if __name__ == "__main__":
    main()
