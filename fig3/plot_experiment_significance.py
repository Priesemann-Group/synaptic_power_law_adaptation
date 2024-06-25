import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statistics import linear_regression

import scipy.stats as ss

plt.rcParams.update({'font.size': 8})

def get_bootstrap_confidence_intervals(data, n_bootstrap=10000, ci=0.95):
    mean = data.mean()
    bootstrapped_means = [np.mean(np.random.choice(data, size=len(data))) for i in range(n_bootstrap)]
    percentage = 100*(1-ci)/2
    lower_err = mean - np.percentile(bootstrapped_means, percentage)
    upper_err = np.percentile(bootstrapped_means, 100-percentage) - mean
    yerrs = [[lower_err], [upper_err]]

    return yerrs
        


def annotate_significance(ax, x1, x2, y_0, p, level_height = 1, line_thickness=0.5):
    one_star = 0.05
    two_stars = 0.005
    three_stars = 0.0005

    if x1 > x2:
        x1, x2 = x2, x1
    level = x2 - x1 - 1
    x = (x1 + x2) / 2
    y = y_0 + level_height * level
    if level == 1 or level == 2:
        y -= 0.06 * level_height * ((-1) ** x1) # alternate up and down correction for level 1
    ax.plot([x1 + 0.1, x2 - 0.1], [y, y], color='black', lw=line_thickness)
    ax.plot([x1 + 0.1, x1 + 0.1], [y-0.3*level_height, y], color='black', lw=line_thickness)
    ax.plot([x2 - 0.1, x2 - 0.1], [y-0.3*level_height, y], color='black', lw=line_thickness)
    if p < one_star and p >= two_stars:
        ax.text(x, y+0.3*level_height, "*", fontsize=8, ha='center', va='center')
    elif p < two_stars and p >= three_stars:
        ax.text(x, y+0.3*level_height, "**", fontsize=8, ha='center', va='center')
    elif p < three_stars:
        ax.text(x, y+0.3*level_height, "***", fontsize=8, ha='center', va='center')


def plot_model(data, name, level_height = 1):
    # get data sorted by column name
    # data = data.loc[:, model.columns]
    data_means = data.mean(axis=0).values
    data_vars = data.var(axis=0).values
    columns = data.columns

    test_data = [data[c].dropna().values for c in columns]
    H, p = ss.kruskal(*test_data)
    print("Kruskal-Wallis H-test p-value: {}".format(p))

    melted_data = data.melt(var_name='condition', value_name='output').dropna()

    tukey_data = [data[c].dropna().values for c in columns]
    t = ss.tukey_hsd(*tukey_data)
    t = pd.DataFrame(t.pvalue, columns=columns, index=columns)
    print(t)

    # plot data and significance
    fig, ax = plt.subplots(1,1, figsize=(4,3))
    x = np.arange(len(data_means))
    # x is category, y is value
    ax.plot(x, data_means, 'o', color='gray')
    # plot significance
    max_y = np.max(data_means)
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            stim_i = columns[i].partition("_")[0]
            stim_j = columns[j].partition("_")[0]
            if stim_i == stim_j:
                p = t.iloc[i,j]
                if p < 0.05:
                    annotate_significance(ax, x[i], x[j], max_y + level_height, p, level_height=level_height)

    # plot bootstrap error bars
    for i, column in enumerate(columns):
        yerrs = get_bootstrap_confidence_intervals(data[column].dropna())
        ax.errorbar(x[i], data[column].mean(), yerr=yerrs, color='gray', capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(columns)
    ax.set_xlabel("condition")
    ax.set_ylabel("mean value")
    plt.tight_layout()
    plt.savefig(name)



# load experimental data
excel_file = "data.xlsx" # change path to the location of your file
df = pd.read_excel(excel_file)
columns = list(df.columns)

# plot significance
# plot_model(df, "significance_data.pdf", level_height=0.005)

# plot qq plots of conditions to check if gaussian
sidelength = int(np.ceil(np.sqrt(len(columns))))
fig, ax = plt.subplots(sidelength, sidelength, figsize=(sidelength*1.3, sidelength*1.3))
ax = ax.flatten()
for i, column in enumerate(columns):
    ss.probplot(df[column].dropna(), plot=ax[i])
    # reduce markersize
    for line in ax[i].lines:
        line.set_markersize(2)
    ax[i].set_title(column)
    ax[i].set_xlabel("")
    ax[i].set_ylabel("")
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].spines['left'].set_visible(False)
    ax[i].grid(False)
plt.tight_layout()
plt.savefig("qq_plots.pdf")

# plot data violin plot
import seaborn as sns
columns = list(df.columns)
data = pd.melt(df, value_vars=columns, var_name="Experiment", value_name="Value")
plt.figure(figsize=(4, 3))
sns.stripplot(x="Experiment", y="Value", data=data, color='black', size=1.0, zorder=1, dodge=True, jitter=0.1)
violin_ax = sns.violinplot(x="Experiment", y="Value", data=data, inner="quart", hue=True,
    hue_order=[True, False], split=True, cut=0, zorder=10, linewidth=1)

plt.xlabel("Condition (stimulus length, pause length)")
plt.ylabel("dF/F change upon test stimulus")
plt.xticks(ticks=range(len(columns)), labels=columns, rotation=45)
sns.despine() # if you want to remove the top and right axis
plt.tight_layout()

fig = violin_ax.get_figure()
fig.savefig("violin_plot.pdf") 
