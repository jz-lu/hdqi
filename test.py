import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# 1. Load the CSVs
df1 = pd.read_csv("../data_hdqi/Stephen_in/gibbs_data.csv")                             # has header: n, m, k, e, ratio, semicircle
df2 = pd.read_csv("../data_hdqi/Stephen_out/SA_Stephen.csv", header=None,              # no header
                  names=["m", "n", "k", "SA"])

# 2. Ensure m, n, k are integers in both
for col in ("m", "n", "k"):
    df1[col] = df1[col].astype(int)
    df2[col] = df2[col].astype(int)

# 3. Merge SA from df2 into df1 on the (m, n, k) keys
df1 = df1.merge(
    df2[["m", "n", "k", "SA"]],
    on=["m", "n", "k"],
    how="left",      # preserves all rows of df1 in original order
    validate="one_to_one"
)

df = df1
# Now df1 has a new column 'SA' aligned correctly.
print(df.head())

import os
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs('plots', exist_ok=True)

# Compute integer ratio column
df['ratio'] = (df['m'] / df['n']).astype(int)

# Get unique integer ratios and sorted k values
ratios = sorted(df['ratio'].unique())
unique_k = sorted(df['k'].unique())

# Prepare a consistent color mapping for k values
cmap = plt.get_cmap('tab10')
colors = {k: cmap(i % 10) for i, k in enumerate(unique_k)}

# Loop over each integer ratio and generate a scatter plot
for r in ratios:
    sub_r = df[df['ratio'] == r]
    if sub_r.empty:
        continue

    fig, ax = plt.subplots()
    # Plot semicircle (triangle) and SA (circle) for each k
    for k in unique_k:
        sub_k = sub_r[sub_r['k'] == k]
        if sub_k.empty:
            continue
        ax.scatter(
            sub_k['n'], sub_k['semicircle'],
            marker='^', label=f'k={k} (semicircle)',
            color=colors[k]
        )
        ax.scatter(
            sub_k['n'], sub_k['SA'],
            marker='o', label=f'k={k} (SA)',
            color=colors[k]
        )

    # Create separate legends
    from matplotlib.lines import Line2D
    # Legend for k (color)
    color_handles = [
        Line2D([0], [0], marker='o', color=colors[k], linestyle='None', markersize=8)
        for k in unique_k if not sub_r[sub_r['k'] == k].empty
    ]
    color_labels = [f'k={k}' for k in unique_k if not sub_r[sub_r['k'] == k].empty]
    legend_k = ax.legend(color_handles, color_labels, title='Value of k', loc='lower left')

    # Legend for marker style (metric)
    style_handles = [
        Line2D([0], [0], marker='^', color='black', linestyle='None', markersize=8),
        Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=8)
    ]
    style_labels = ['semicircle (triangle)', 'SA (circle)']
    legend_style = ax.legend(style_handles, style_labels, title='Metric', loc='lower right')
    ax.add_artist(legend_k)

    # Labels, limits, title
    ax.set_xlabel('n')
    ax.set_ylabel('Metric Value')
    ax.set_ylim(0.4, 0.8)
    ax.set_title(f'Ratio m/n = {r}')

    # Save the figure
    filename = f'plots/ratio_{r}.png'
    plt.savefig(filename)
    plt.close(fig)

print(f"Generated {len(ratios)} scatter plots in the 'plots' directory.")
