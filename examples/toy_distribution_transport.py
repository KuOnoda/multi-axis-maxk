#!/usr/bin/env python3
"""CPU-only synthetic distribution-transport example from the paper.

One unified D=9 toy (3x3 grid of 2D Gaussian group-modes), three views:

  (A) 2D before/after (2x2): the generator's sampling cloud starts skewed on a
      few majority modes (before=base) and multi-axis max@K TRANSPORTS the mass
      to cover/balance all 9 groups (after); a scalar reward instead COLLAPSES
      it onto one mode. Color = group, so coverage/balance reads at a glance.
  (B) K = how far you can move it: final balance vs the training group size K
      (K=1 has no multi-sample credit and leaves the initial policy unchanged).
  (C) M = i.i.d. readout: occurrence of a rare group = 1-(1-p)^M of the MOVED
      marginal (no anti-i.i.d. claim; occurrence follows from the marginal).
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# ----- D=9 2D mode layout -----
SP = 2.6
CENTERS = np.array([[i * SP, j * SP] for j in (1, 0, -1) for i in (-1, 0, 1)], float)
D = len(CENTERS)
SIGMA = 0.34
EPS = 1e-9
N_SHOW = 2400
AFTER = 6
STEPS = 60
LR = 0.45
MCMAP = plt.get_cmap("tab10")
MODE_COLORS = [MCMAP(i % 10) for i in range(D)]

# skewed base: majority corner + a little center/top-right
W0 = np.full(D, 0.012)
W0[0] = 0.80
W0[4] = 0.10
W0[2] = 0.05
W0 = W0 / W0.sum()
QUAL = np.zeros(D)
QUAL[0] = 1.0  # scalar reward prefers the majority mode


def softmax(z):
    z = z - z.max()
    values = np.exp(z)
    return values / values.sum()


def adv_maxk(types):
    counts = np.bincount(types, minlength=D)
    return np.array([1.0 if counts[t] == 1 else 0.0 for t in types])


def adv_scalar(types):
    scores = QUAL[types]
    return scores - scores.mean()


def train(advfn, K, steps, seed, lr=LR):
    rng = np.random.default_rng(seed)
    theta = np.log(W0 + EPS)
    trajectory = [softmax(theta).copy()]
    for _ in range(steps):
        weights = softmax(theta)
        gradient = np.zeros(D)
        for _g in range(400):
            types = rng.choice(D, size=K, p=weights)
            advantage = advfn(types)
            advantage = advantage - advantage.mean()
            std = advantage.std()
            if std > 1e-8:
                advantage = advantage / std
            for index, mode in enumerate(types):
                one_hot = np.zeros(D)
                one_hot[mode] = 1.0
                gradient += advantage[index] * (one_hot - weights)
        theta = theta + lr * gradient / 400
        trajectory.append(softmax(theta).copy())
    return np.array(trajectory)


def balance(w):
    return 1 - 0.5 * np.abs(w - 1 / D).sum() / (1 - 1 / D)


def cloud(w, seed=123):
    rng = np.random.default_rng(seed)
    m = rng.choice(D, size=N_SHOW, p=w)
    return CENTERS[m] + rng.normal(scale=SIGMA, size=(N_SHOW, 2)), m


def draw_cloud(ax, w):
    for center, mode_color in zip(  # noqa: B905 - also runs in the local Python 3.9 shell
        CENTERS, MODE_COLORS
    ):
        ax.add_patch(
            plt.Circle(
                center,
                SIGMA * 2.4,
                fill=False,
                ec=mode_color,
                lw=1.6,
                alpha=0.6,
                zorder=1,
            )
        )
    points, modes = cloud(w)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=7,
        c=[MODE_COLORS[index] for index in modes],
        alpha=0.5,
        edgecolors="none",
        zorder=2,
        rasterized=True,
    )
    ax.set_xlim(-SP * 1.42, SP * 1.42)
    ax.set_ylim(-SP * 1.42, SP * 1.42)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("0.75")


def main():
    tr_m = train(adv_maxk, K=16, steps=STEPS, seed=1)
    tr_s = train(adv_scalar, K=16, steps=STEPS, seed=2)
    print(
        f"max@K balance: base {balance(tr_m[0]):.3f}  after(step{AFTER}) {balance(tr_m[AFTER]):.3f}"
    )
    print(
        f"scalar balance: base {balance(tr_s[0]):.3f}  "
        f"after(step{AFTER}) {balance(tr_s[AFTER]):.3f}"
    )

    # Panel B: balance vs K
    Ks = [1, 2, 4, 8, 12, 16]
    balB = [balance(train(adv_maxk, K=K, steps=STEPS, seed=30 + K)[-1]) for K in Ks]

    # Panel C: rare-group occupancy readout
    rare = int(np.argmin(W0))
    p_base, p_moved = W0[rare], tr_m[AFTER][rare]
    Ms = np.arange(1, 17)

    # ---------------- figure ----------------
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(16.0, 5.4))
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.0, 1.3, 1.3],
        wspace=0.24,
        left=0.035,
        right=0.99,
        top=0.84,
        bottom=0.13,
    )

    # (A) 2x2 before/after — packed tight
    gsA = outer[0].subgridspec(2, 2, wspace=0.05, hspace=0.05)
    axes = [[fig.add_subplot(gsA[r, c]) for c in (0, 1)] for r in (0, 1)]
    draw_cloud(axes[0][0], tr_m[0])
    draw_cloud(axes[0][1], tr_m[AFTER])
    draw_cloud(axes[1][0], tr_s[0])
    draw_cloud(axes[1][1], tr_s[AFTER])
    axes[0][0].set_title("before", fontsize=16)
    axes[0][1].set_title("after", fontsize=16)
    axes[0][0].set_ylabel("multi-axis\nmax@K (ours)", fontsize=15)
    axes[1][0].set_ylabel("scalar\nreward", fontsize=15)
    # A region center for the group title
    aw = 2.0 / (2.0 + 1.3 + 1.3)
    a_left, a_right = 0.035, 0.035 + aw * (0.99 - 0.035)
    fig.text(
        (a_left + a_right) / 2,
        0.92,
        "(A) The sampling distribution moves to cover all groups",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    # (B) balance vs K
    axB = fig.add_subplot(outer[1])
    axB.plot(Ks, balB, "o-", color="#1f77b4", lw=3.2, ms=9)
    axB.axhline(balance(W0), ls="--", color="0.4", lw=1.8)
    axB.text(
        Ks[-1],
        balance(W0) + 0.03,
        "base",
        fontsize=12,
        ha="right",
        va="bottom",
        color="0.4",
    )
    axB.set_xlabel("training group size  K")
    axB.set_ylabel("balance after training")
    axB.set_title("(B) K = how far you move it", fontweight="bold")
    axB.set_ylim(0, 1.05)
    axB.grid(alpha=0.3)

    # (C) occupancy readout
    axC = fig.add_subplot(outer[2])
    axC.plot(
        Ms,
        1 - (1 - p_base) ** Ms,
        "--",
        color="#9467bd",
        lw=3.2,
        label=f"base  (p={p_base:.2f})",
    )
    axC.plot(
        Ms,
        1 - (1 - p_moved) ** Ms,
        "-",
        color="#2ca02c",
        lw=3.4,
        label=f"after max@K  (p={p_moved:.2f})",
    )
    axC.set_xlabel("evaluation budget  M")
    axC.set_ylabel("P(rare group in M)")
    axC.set_title("(C) M : readout $1-(1-p)^M$", fontweight="bold")
    axC.set_ylim(0, 1.05)
    axC.grid(alpha=0.3)
    axC.legend(fontsize=12, loc="upper left", framealpha=0.9)

    fig.savefig(f"{OUT}/toy_distribution_transport.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/toy_distribution_transport.png", dpi=200, bbox_inches="tight")
    summary = {
        "balance_maxk_after": balance(tr_m[AFTER]),
        "balance_scalar_after": balance(tr_s[AFTER]),
        "balance_base": balance(W0),
        "Ks": Ks,
        "balB": balB,
        "rare_p_base": float(p_base),
        "rare_p_moved": float(p_moved),
        "after": AFTER,
    }
    with open(f"{OUT}/toy_summary.json", "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    print("WROTE", f"{OUT}/toy_distribution_transport.png")


if __name__ == "__main__":
    main()
