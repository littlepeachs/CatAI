"""Generate donut charts for metal distribution and heterogeneous support distribution."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_PATH = Path("./coupling_info_results.json")
OUTPUT_PATH = Path("./metal_support_donut.png")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------------------------------------------------------------------------
# Count metals (all records) and supports (heterogeneous only)
# ---------------------------------------------------------------------------
metal_counter: Counter = Counter()
support_counter: Counter = Counter()

for rec in data:
    metals = (rec.get("metal") or "").split("/")
    for m in metals:
        m = m.strip()
        if m and m.lower() != "none":
            metal_counter[m] += 1
    if rec.get("catalysis_type") == "heterogeneous":
        s = rec.get("support") or "Others"
        if s.lower() == "none":
            s = "Others"
        support_counter[s] += 1

# ---------------------------------------------------------------------------
# Prepare metal data: top 6 + Others
# ---------------------------------------------------------------------------
TOP_N_METALS = 6
metal_items = metal_counter.most_common()
top_metals = metal_items[:TOP_N_METALS]
others_count = sum(v for _, v in metal_items[TOP_N_METALS:])
metal_labels = [k for k, _ in top_metals] + ["Others"]
metal_values = [v for _, v in top_metals] + [others_count]

# ---------------------------------------------------------------------------
# Prepare support data (fixed order)
# ---------------------------------------------------------------------------
SUPPORT_ORDER = ["Metal Oxides", "Carbon", "Nitride", "Metals", "Others"]
support_labels = SUPPORT_ORDER
support_values = [support_counter.get(s, 0) for s in SUPPORT_ORDER]

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
metal_colors = [
    '#C8E6FF',  # 1st
    '#99CCFF',  # 2nd
    '#66B2FF',  # 3rd
    '#E8F5E8',  # 4th
    '#D0EBD0',  # 5th
    '#B8E0B8',  # 6th
    '#D8D8D8',  # Others
]

support_colors = [
    '#C8E6FF',  # Metal Oxides
    '#99CCFF',  # Carbon
    '#66B2FF',  # Nitride
    '#B8E0B8',  # Metals
    '#D8D8D8',  # Others
]

# ---------------------------------------------------------------------------
# Helper: draw one styled donut on an axes
# ---------------------------------------------------------------------------
THRESHOLD = 5.0  # percent threshold: show label inside if >= this value

def draw_donut(ax, values, labels, colors, center_label, title):
    total = sum(values)
    pct_values = [v / total * 100 for v in values]

    wedges, _ = ax.pie(
        values,
        startangle=90,
        counterclock=False,
        colors=colors[:len(labels)],
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white', 'width': 0.5},
    )

    # Centre hole label
    ax.text(0, 0, center_label, ha='center', va='center',
            fontsize=22, fontweight='bold', fontname='Arial', color='#333333')

    # Percentage labels inside large wedges
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        if pct_values[i] >= THRESHOLD:
            ax.text(x * 0.75, y * 0.75, f"{pct_values[i]:.1f}%",
                    fontsize=22, fontweight='bold', fontname='Arial',
                    color='black', ha='center', va='center')

    # Legend on the right of each subplot
    ax.legend(
        wedges, labels,
        title="",
        loc="center left",
        bbox_to_anchor=(0.9, 0, 0.5, 1),
        frameon=False,
        fontsize=14,
        prop={'family': 'Arial', 'size': 14},
        handlelength=1.0,
        handleheight=1.0,
    )

    ax.set_title(title, fontsize=14, pad=12, fontname='Arial')
    ax.set_aspect('equal')


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

draw_donut(
    ax1, metal_values, metal_labels, metal_colors,
    center_label="Metal",
    title=f"Metal Distribution (n={sum(metal_values)})",
)

draw_donut(
    ax2, support_values, support_labels, support_colors,
    center_label="Support",
    title=f"Heterogeneous Support (n={sum(support_values)})",
)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved to {OUTPUT_PATH}")
print(f"Saved to {OUTPUT_PATH}")
