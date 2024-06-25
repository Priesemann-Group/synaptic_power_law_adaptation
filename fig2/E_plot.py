import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys, os, shutil


plt.rcParams.update({'font.size': 8})


filename = "model_fit.h5"

print("Plotting " + filename)
dic = h5py.File(filename, 'r')
folder = "."
#shutil.rmtree(folder, ignore_errors=True)

try:
    os.mkdir(folder)
except:
    pass


bs = dic.get('b')[()]
kappas = dic.get('kappa')[()]
timebins = dic.get('timebins')[()]
loglikelihood = dic.get('loglikelihood')[()]
bs_single = dic.get('b_single')[()]
kappas_single = dic.get('kappa_single')[()]
timebins_single = dic.get('timebins_single')[()]
loglikelihood_single = dic.get('loglikelihood_single')[()]



fig, axs = plt.subplots(1, 3, figsize=(3*2.1,1.6))#, sharey=True)


axs[0].plot(loglikelihood[:-1])
axs[0].set_ylabel(r'loglikelihood')
axs[0].set_xlabel(r'iteration')
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)

axs[1].plot(bs)
axs[1].set_ylabel(r'b')
axs[1].set_xlabel(r'iteration')
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)

steps = 20
for i in np.arange(1, kappas.shape[1], int(kappas.shape[1]/steps)):
    axs[2].plot(timebins, -kappas[:,i-1])
axs[2].set_ylabel(r'$\kappa (\Delta t)$')
axs[2].set_xlabel(r'$\Delta t$ [s]')
axs[2].set_yscale('log')
axs[2].set_xscale('log')
axs[2].spines["top"].set_visible(False)
axs[2].spines["right"].set_visible(False)

import matplotlib.cm as cm

cmap = cm.get_cmap('viridis')
colors = [cmap(1 - i/steps) for i in range(steps)]
for i,j in enumerate(axs[2].lines):
    j.set_color(colors[i])
    j.set_label('iteration {}'.format(i))

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=steps))
sm.set_array([])
cbar = plt.colorbar(sm, ax=axs[2], orientation='vertical', ticks=[0, 10, 20])
cbar.ax.set_yticklabels(['20', '10', '0'])
cbar.set_label('iteration')

plt.tight_layout()
plt.savefig("model_fit.pdf")


# fit kernel with truncated power law

from scipy.optimize import curve_fit

cutoff_ind1 = 10 # cut off really noisy estimates 
cutoff_ind2 = -1 # cut off really noisy estimates
cutoff_ind2single = -10 # cut off really noisy estimates
x = np.log(timebins[cutoff_ind1:cutoff_ind2])
y = np.log(-kappas[cutoff_ind1:cutoff_ind2,-1])
y2 = -kappas_single[cutoff_ind1:cutoff_ind2,-1]

def piecewise_linear(x, t, b, m, m2):
    return np.piecewise(x, [x < t, x >= t], [lambda x: m * x + b, lambda x: m * t + b + m2 * (x - t)])

# Define initial guesses for parameters
p0 = [2, 0, -1, -10]

# Fit model
popt, pcov = curve_fit(piecewise_linear, x, y, p0=p0)
p_log = lambda x: piecewise_linear(x, *popt)

t, b, m, m2 = popt

print("t={:.1f}, b={:.1f}, m={:.1f}, m2={:.1f}".format(*popt))
print("kappa(dt) ~ dt^{:.1f} if dt < {:.1f}".format(m,np.exp(t)))
print("kappa(dt) ~ dt^{:.1f} if dt >= {:.1f}".format(m2,np.exp(t)))

# transform back to original scale
x = np.exp(x)
y = np.exp(y)
p = lambda x: np.exp(p_log(np.log10(x)))

# Plot the data with the best-fit model
fig, ax = plt.subplots(1, 1, figsize=(2.1,1.6))#, sharey=True)
# plot as step plot
#ax.step(x, y, color='red', label='kernel')
ax.plot(x[:cutoff_ind2single], y2[:cutoff_ind2single], color='gray', label='kernel single')
ax.plot(x, y, color='red', label='kernel')
ax.plot(x, p(x), color='black', linestyle='--', linewidth=0.5, label=r'${:.1f} * x^{{{:.1f}}} {:.2f}$'.format(*popt))
ax.set_ylabel(r'$\kappa (\Delta t)$')
ax.set_xlabel(r'$\Delta t$ [s]')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([1e-3, 10e-1])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ticks at all powers of 10
ax.set_xticks([0.1,1, 10, 100])
mticker = plt.matplotlib.ticker
ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
#ax.legend(frameon=False)

axsins = ax.inset_axes([0.1, 0.65, 0.4, 0.4])
axsins.plot(x-1, 0*x, color='black', linestyle='--', linewidth=0.3, label=r'${:.1f} * x^{{{:.1f}}} {:.2f}$'.format(*popt))
axsins.plot(x, y2, color='gray', label='kernel single')
axsins.plot(x, y, color='red', label='kernel')
axsins.set_xlim([-0.5, 8])
axsins.set_ylim([-0.05, 0.25])
axsins.spines["top"].set_visible(False)
axsins.spines["right"].set_visible(False)
# remove ticks
axsins.set_xticks([])
axsins.set_yticks([])



plt.tight_layout()
plt.savefig("model_kernel_fit.pdf")


# fig, axs = plt.subplots(1, 1, figsize=(1*2.1,1.6))#, sharey=True)

# kappa = - np.hstack([kappas[1,-1],kappas[:,-1],0])
# timebins = np.hstack([0, timebins, timebins[-1]])

# axs.plot(timebins, kappa, color='black')
# axs.set_ylabel(r'$\kappa (\Delta t)$')
# axs.set_xlabel(r'$\Delta t$ [s]')
# axs.set_yscale('log')
# axs.set_xscale('log')
# axs.spines["top"].set_visible(False)
# axs.spines["right"].set_visible(False)

# plt.tight_layout()
# plt.savefig("../out/plots/model_kernel.pdf")


