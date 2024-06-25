import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from tqdm.auto import tqdm

plt.rcParams.update({'font.size': 8})

fm = lambda x: round(x, 1) if x % 1 else int(x)

def bootstrap_confidence_interval(data, n=1000, func=np.mean, alpha=0.05):
    resamples = np.random.choice(data.dropna(), size=(n, len(data.dropna())), replace=True)
    perc = np.percentile([func(r) for r in resamples], [100*alpha/2, 100*(1-alpha/2)])
    return [func(data.dropna()) - perc[0], perc[1] - func(data.dropna())]

def percentile_confidence_interval(data, alpha=0.05):
    perc = np.percentile(data.dropna(), [100*alpha/2, 100*(1-alpha/2)])
    return [data.mean() - perc[0], perc[1] - data.mean()]

def get_posterior_samples(trace):
    sim_df = pd.DataFrame()
    for condition in conditions:
        depletiontime, pausetime = condition
        condition_name = "{}_{}".format(fm(depletiontime), fm(pausetime))
        sim_df[condition_name] = trace.posterior[condition_name].to_numpy().flatten()
    return sim_df

def plot_posterior_comparison(sim_df, df, color, name):

    fig = plt.figure(figsize=(3,2))
    ax = fig.subplots(1,1)

    # remove the right and top and bottom spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # compute conf. int. for samples from the posterior means
    errs_sim_df = sim_df.apply(lambda x: percentile_confidence_interval(x), axis=0)
    # compute conf. int. for the data
    errs_df = df.apply(lambda x: bootstrap_confidence_interval(x), axis=0)

    lin = plt.errorbar(df.columns, df.mean(), yerr=errs_df, linestyle='None', marker='None', markersize=2, elinewidth=9, color='black', alpha=0.2)
    mark = plt.errorbar(df.columns, df.mean(), yerr=errs_df, linestyle='None', marker='_', markersize=9, elinewidth=0, color='black', alpha=0.2)
    sim = plt.errorbar(sim_df.columns, sim_df.mean(), yerr=errs_sim_df, linestyle='None', marker='o', color=color, markersize=5)

    namedic = {
        'full_model': 'full model',
        'single_timescale': 'single ts. model',
        'two_timescales': 'two ts. model'
    }

    ax.legend([(lin, mark), sim], ['data', namedic[name]], frameon=False)
    plt.xticks(rotation=45)
    plt.ylabel("Change in dF/F")
    plt.savefig("posterior_comparison_{}.pdf".format(name), bbox_inches='tight')


def plot_data(df):
    fig = plt.figure(figsize=(3.5,2.8))
    ax = fig.subplots(1,1)

    # remove the right and top and bottom spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # compute conf. int. for the data
    errs_df = df.apply(lambda x: bootstrap_confidence_interval(x), axis=0)

    lin = plt.errorbar(df.columns, df.mean(), yerr=errs_df, linestyle='None', marker='None', markersize=2, elinewidth=9, color='black', alpha=0.2)
    mark = plt.errorbar(df.columns, df.mean(), yerr=errs_df, linestyle='None', marker='_', markersize=9, elinewidth=0, color='black', alpha=0.2)

    plt.xticks(rotation=45)
    plt.ylabel("Change in dF/F")
    plt.savefig("data.pdf", bbox_inches='tight')

# get experimental data
excel_file = "data.xlsx" # change path to the location of your file
df = pd.read_excel(excel_file)

depletiontimes = [0.4, 4.0, 40.0]
pausetimes = [1.0, 10.0, 40.0, 100.0]
conditions = [(depletiontime, pausetime) for pausetime in pausetimes for depletiontime in depletiontimes]
#conditions.append((40.0, 200.0))

# load traces
trace = az.from_netcdf("trace.nc")
trace_single = az.from_netcdf("trace_single.nc")
trace_tt = az.from_netcdf("trace_tt.nc")

# get the posterior
sim_df = get_posterior_samples(trace)
sim_df_single = get_posterior_samples(trace_single)
sim_df_tt = get_posterior_samples(trace_tt)

plot_posterior_comparison(sim_df, df, 'cornflowerblue', 'full_model')
plot_posterior_comparison(sim_df_single, df, 'gray', 'single_timescale')
plot_posterior_comparison(sim_df_tt, df, 'gray', 'two_timescales')

plot_data(df)
