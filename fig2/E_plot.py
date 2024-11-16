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
loglikelihood_test = dic.get('loglikelihood_test')[()]
bs_single = dic.get('b_single')[()]
kappas_single = dic.get('kappa_single')[()]
timebins_single = dic.get('timebins_single')[()]
loglikelihood_single = dic.get('loglikelihood_single')[()]
loglikelihood_test_single = dic.get('loglikelihood_test_single')[()]


def plot_kernel(kappas, timebins, likelihood, likelihood_test, best_step, filename):
    fig, axs = plt.subplots(1, 3, figsize=(3*2.1,1.6))#, sharey=True)

    axs[0].plot(likelihood[:-1], label='train')
    axs[0].plot(likelihood_test[:-1], label='test')
    axs[0].set_ylabel(r'loglikelihood')
    axs[0].set_xlabel(r'iteration')
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].legend(frameon=False)

    axs[1].plot(bs)
    axs[1].set_ylabel(r'b')
    axs[1].set_xlabel(r'iteration')
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    for i in range(best_step,0,-1):
        axs[2].plot(timebins, -kappas[:,best_step - i])
    axs[2].set_ylabel(r'$\kappa (\Delta t)$')
    axs[2].set_xlabel(r'$\Delta t$ [s]')
    axs[2].set_yscale('log')
    axs[2].set_xscale('log')
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)

    import matplotlib.cm as cm

    cmap = cm.get_cmap('viridis')
    colors = [cmap((best_step-i)/best_step) for i in range(best_step)]
    for i,j in enumerate(axs[2].lines):
        j.set_color(colors[i])
        j.set_label('iteration {}'.format(best_step - i))

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=best_step, vmax=0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axs[2], orientation='vertical') #, ticks=[0, 5, 10])
    # cbar.ax.set_yticklabels(['10', '5', '0'])
    cbar.set_label('iteration')

    plt.tight_layout()
    plt.savefig(filename)

best_step = len(bs)-1
plot_kernel(kappas, timebins, loglikelihood, loglikelihood_test, best_step, "model_fit.pdf")
best_step_single = len(bs_single)-1
plot_kernel(kappas_single, timebins_single, loglikelihood_single, loglikelihood_test_single, best_step_single, "model_fit_single.pdf")


# try fitting several models

from scipy.optimize import curve_fit

cutoff_ind1 = 10 # cut off really noisy estimates 
cutoff_ind2 = -1 # cut off really noisy estimates
cutoff_ind2single = -5 # cut off really noisy estimates
x = timebins[cutoff_ind1:cutoff_ind2]
y = -kappas[cutoff_ind1:cutoff_ind2,best_step]
y2 = -kappas_single[cutoff_ind1:cutoff_ind2,best_step]

x_log = np.log(x)
y_log = np.log(y)


# fit kernel with truncated power law

def piecewise_linear(x_log, t, b, m, m2):
    return np.piecewise(x_log, [x_log < t, x_log >= t], 
                        [lambda x_log: m * x_log + b, 
                         lambda x_log: m * t + b + m2 * (x_log - t)])

# Define initial guesses for parameters
p0 = [1, 0, -1, -10]

# Fit model
try:
    popt, pcov = curve_fit(piecewise_linear, x_log, y_log, p0=p0)
    p_log = lambda x: piecewise_linear(x, *popt)
except:
    print("full model piecewise linear fit failed")
    popt = p0
    p_log = lambda x: piecewise_linear(x, *popt)

t, b, m, m2 = popt

print("truncated power law fit")
print("t={:.1f}, b={:.1f}, m={:.1f}, m2={:.1f}".format(*popt))
print("kappa(dt) ~ dt^{:.1f} if dt < {:.1f}".format(m,np.exp(t)))
print("kappa(dt) ~ dt^{:.1f} if dt >= {:.1f}".format(m2,np.exp(t)))

p = lambda x: np.exp(p_log(np.log(x)))



# fit lognormal kernel

def log_lognormal(x_log, a, b, c):
    return np.log(a) - 0.5 * ((x_log - b) / c)**2

p0_lognormal = [1, 0, 1]
try:
    popt_lognormal, pcov_lognormal = curve_fit(log_lognormal, x_log, y_log, p0=p0_lognormal)
except:
    print("full model lognormal fit failed")
    popt_lognormal = p0_lognormal

a, b, c = popt_lognormal

print("lognormal fit")
print("a={:.1f}, b={:.1f}, c={:.1f}".format(*popt_lognormal))

p_lognormal = lambda x: np.exp(log_lognormal(np.log(x), *popt_lognormal))


# fit exponential kernel

def exponential(x, a, b):
    return np.log(a) + (-b * x)

p0_exponential = [1, 1]
try:
    popt_exponential, pcov_exponential = curve_fit(exponential, x, y_log, p0=p0_exponential)
except:
    print("full model exponential fit failed")
    popt_exponential = p0_exponential
a, b = popt_exponential

print("exponential fit")
print("a={:.1f}, b={:.3f}".format(*popt_exponential))

p_exponential = lambda x: np.exp(exponential(x, *popt_exponential))


# fit stretched exponential kernel

def log_stretched_exponential(x, a, b, c):
    return np.log(a) + (-b * x**c) # not using log(x) here!

p0_stretched_exponential = [1, 1, 1]
try:
    popt_stretched_exponential, pcov_stretched_exponential = curve_fit(log_stretched_exponential, x, y_log, p0=p0_stretched_exponential)
except:
    print("full model str. exponential fit failed")
    popt_stretched_exponential = p0_stretched_exponential
a, b, c = popt_stretched_exponential

print("stretched exponential fit")
print("a={:.1f}, b={:.1f}, c={:.1f}".format(*popt_stretched_exponential))

p_stretched_exponential = lambda x: np.exp(log_stretched_exponential(x, *popt_stretched_exponential))



# compute AIC and BIC

def AIC(k, L):
    return 2*k - 2*L + 2*k*(k+1)/(len(y) - k - 1)

def BIC(k, L, n):
    return k * np.log(n) - 2*L

# see Model Selection and Multimodel Inference by Burnham and Anderson, 1.2.1
def log_likelihood(y, y_pred):
    var = np.var(y - y_pred)
    return -0.5 * len(y) * np.log(var)

n = len(y)
k = len(p0)
L = log_likelihood(y, p(x))
AIC_truncated_power_law = AIC(k, L)
BIC_truncated_power_law = BIC(k, L, n)

k = len(p0_lognormal)
L = log_likelihood(y, p_lognormal(x))
AIC_lognormal = AIC(k, L)
BIC_lognormal = BIC(k, L, n)

k = len(p0_exponential)
L = log_likelihood(y, p_exponential(x))
AIC_exponential = AIC(k, L)
BIC_exponential = BIC(k, L, n)

k = len(p0_stretched_exponential)
L = log_likelihood(y, p_stretched_exponential(x))
AIC_stretched_exponential = AIC(k, L)
BIC_stretched_exponential = BIC(k, L, n)

print("AIC")
print("truncated power law: {:.1f}".format(AIC_truncated_power_law))
print("lognormal: {:.1f}".format(AIC_lognormal))
print("exponential: {:.1f}".format(AIC_exponential))
print("stretched exponential: {:.1f}".format(AIC_stretched_exponential))

print("BIC")
print("truncated power law: {:.1f}".format(BIC_truncated_power_law))
print("lognormal: {:.1f}".format(BIC_lognormal))
print("exponential: {:.1f}".format(BIC_exponential))
print("stretched exponential: {:.1f}".format(BIC_stretched_exponential))





# fit single kernel with truncated power law

x_single = timebins_single[cutoff_ind1:cutoff_ind2single]
y_single = -kappas_single[cutoff_ind1:cutoff_ind2single,best_step]

x_log_single = np.log(x_single)
y_log_single = np.log(y_single)


# fit kernel with truncated power law
p0 = [1, 0, -1, -3]

try:
    popt_single, pcov_single = curve_fit(piecewise_linear, x_log_single, y_log_single, p0=p0)
except:
    print("single ts model piecewise linear fit failed")
    popt_single = p0
t, b, m, m2 = popt_single

print("truncated power law fit single")
print("t={:.1f}, b={:.1f}, m={:.1f}, m2={:.1f}".format(*popt_single))

p_single = lambda x: np.exp(piecewise_linear(np.log(x), *popt_single))


# fit lognormal kernel

try:
    popt_lognormal_single, pcov_lognormal_single = curve_fit(log_lognormal, x_log_single, y_log_single, p0=p0_lognormal)
except:
    print("single ts model lognormal fit failed")
    popt_lognormal_single = p0_lognormal
a, b, c = popt_lognormal_single

print("lognormal fit single")
print("a={:.1f}, b={:.1f}, c={:.1f}".format(*popt_lognormal_single))

p_lognormal_single = lambda x: np.exp(log_lognormal(np.log(x), *popt_lognormal_single))


# fit exponential kernel
def exponential(x, a, b):
    return a*np.exp(-b*x)

try:
    popt_exponential_single, pcov_exponential_single = curve_fit(exponential, x_single, y_single, p0=p0_exponential)
except:
    print("single ts model exponential fit failed")
    popt_exponential_single = p0_exponential
a, b = popt_exponential_single

print("exponential fit single")
print("a={:.1f}, b={:.3f}".format(*popt_exponential_single))

p_exponential_single = lambda x: exponential(x, *popt_exponential_single)


# fit stretched exponential kernel

try:
    popt_stretched_exponential_single, pcov_stretched_exponential_single = curve_fit(log_stretched_exponential, x_single, y_log_single, p0=p0_stretched_exponential)
except:
    print("single ts model str. exponential fit failed")
    popt_stretched_exponential_single = p0_stretched_exponential
a, b, c = popt_stretched_exponential_single

print("stretched exponential fit single")
print("a={:.1f}, b={:.1f}, c={:.1f}".format(*popt_stretched_exponential_single))

p_stretched_exponential_single = lambda x: np.exp(log_stretched_exponential(x, *popt_stretched_exponential_single))


# compute AIC and BIC

n = len(y2)
k = len(p0)
L = log_likelihood(y2, p_single(x))
AIC_truncated_power_law_single = AIC(k, L)
BIC_truncated_power_law_single = BIC(k, L, n)

k = len(p0_lognormal)
L = log_likelihood(y2, p_lognormal_single(x))
AIC_lognormal_single = AIC(k, L)
BIC_lognormal_single = BIC(k, L, n)

k = len(p0_exponential)
L = log_likelihood(y2, p_exponential_single(x))
AIC_exponential_single = AIC(k, L)
BIC_exponential_single = BIC(k, L, n)

k = len(p0_stretched_exponential)
L = log_likelihood(y2, p_stretched_exponential_single(x))
AIC_stretched_exponential_single = AIC(k, L)
BIC_stretched_exponential_single = BIC(k, L, n)

print("AIC single")
print("truncated power law: {:.1f}".format(AIC_truncated_power_law_single))
print("lognormal: {:.1f}".format(AIC_lognormal_single))
print("exponential: {:.1f}".format(AIC_exponential_single))
print("stretched exponential: {:.1f}".format(AIC_stretched_exponential_single))

print("BIC single")
print("truncated power law: {:.1f}".format(BIC_truncated_power_law_single))
print("lognormal: {:.1f}".format(BIC_lognormal_single))
print("exponential: {:.1f}".format(BIC_exponential_single))
print("stretched exponential: {:.1f}".format(BIC_stretched_exponential_single))




# Plot the data with the best-fit model
fig, ax = plt.subplots(1, 1, figsize=(2.1,1.6))#, sharey=True)
# ax.plot(x[:cutoff_ind2single], y2[:cutoff_ind2single], color='gray', label='kernel single')
ax.plot(x, y, color='red', label='kernel')
t, b, m, m2 = popt
ax.plot(x, p(x), color='black', linestyle='--', linewidth=0.5, label=r'${:.1f}x^{{ {:.1f} }}$'.format(np.exp(b),m))
ax.plot(x, p_lognormal(x), color='blue', linestyle='--', linewidth=0.5, label=r'${:.1f} \exp(-0.5 \left(\frac{{\log(x) - {:.1f}}}{{ {:.1f} }}\right)^2)$'.format(*popt_lognormal))
ax.plot(x, p_stretched_exponential(x), color='orange', linestyle='--', linewidth=0.5, label=r'${:.1f} \exp(-{:.1f} x^{{{:.1f}}})$'.format(*popt_stretched_exponential))
# doesn't fit well
# ax.plot(x, p_exponential(x), color='green', linestyle='--', linewidth=0.5, label=r'${:.1f} \exp(-{:.2f} x)$'.format(*popt_exponential))
ax.set_ylabel(r'$\kappa (\Delta t)$')
ax.set_xlabel(r'$\Delta t$ [s]')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([5e-3, 10e-1])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ticks at all powers of 10
ax.set_xticks([0.1,1, 10, 100])
mticker = plt.matplotlib.ticker
ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
ax.legend(frameon=False)

axsins = ax.inset_axes([0.1, 0.65, 0.4, 0.4])
axsins.plot(x-5, 0*x, color='black', linestyle='-', linewidth=0.2)
axsins.plot(x, y2, color='gray', label='kernel single')
axsins.plot(x, y, color='red', label='kernel')
axsins.plot(x, p(x), color='black', linestyle='--', linewidth=0.5, label=r'${:.1f} * x^{{{:.1f}}} {:.2f}$'.format(*popt))
axsins.set_xlim([-5, 80])
axsins.set_ylim([-0.05, 0.25])
axsins.spines["top"].set_visible(False)
axsins.spines["right"].set_visible(False)
# remove ticks
axsins.set_xticks([])
axsins.set_yticks([])

plt.tight_layout()
plt.savefig("model_kernel_fit.pdf")


# Plot the data with the best-fit model
fig, ax = plt.subplots(1, 1, figsize=(2.1,1.6))#, sharey=True)
ax.plot(x_single[:cutoff_ind2single], y_single[:cutoff_ind2single], color='gray', label='kernel single')
ax.plot(x_single, y_single, color='red', label='kernel')
ax.plot(x_single, p_single(x_single), color='black', linestyle='--', linewidth=0.5, label=r'${:.1f}x^{{ {:.1f} }}$'.format(b,m))
ax.plot(x_single, p_lognormal_single(x_single), color='blue', linestyle='--', linewidth=0.5, label=r'${:.1f} \exp(-0.5 \left(\frac{{\log(x) - {:.1f}}}{{ {:.1f} }}\right)^2)$'.format(*popt_lognormal_single))
ax.plot(x_single, p_exponential_single(x_single), color='green', linestyle='--', linewidth=0.5, label=r'${:.1f} \exp(-{:.2f} x)$'.format(*popt_exponential_single))
ax.plot(x_single, p_stretched_exponential_single(x_single), color='orange', linestyle='--', linewidth=0.5, label=r'${:.1f} \exp(-{:.1f} x^{{{:.1f}}})$'.format(*popt_stretched_exponential_single))
ax.set_ylabel(r'$\kappa (\Delta t)$')
ax.set_xlabel(r'$\Delta t$ [s]')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([5e-3, 10e-1])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# ticks at all powers of 10
ax.set_xticks([0.1,1, 10, 100])
mticker = plt.matplotlib.ticker
ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("model_kernel_fit_single.pdf")


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


