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
print("t={:.2f}, b={:.2f}, m={:.2f}, m2={:.2f}".format(*popt))
print("kappa(dt) ~ dt^{:.1f} if dt < {:.1f}".format(m,np.exp(t)))
print("kappa(dt) ~ dt^{:.1f} if dt >= {:.1f}".format(m2,np.exp(t)))

p = lambda x: np.exp(p_log(np.log(x)))


# fit kernel with double exponential

def log_double_exponential(x, a, b, c, d):
    return np.log(a*np.exp(-x/b) + c*np.exp(-x/d))

# Define initial guesses for parameters
p0_double_exponential = [0.1, 0.01, 0.01, 1.0]

# Fit model
try:
    popt_double_exponential, pcov = curve_fit(log_double_exponential, x, y_log, p0=p0_double_exponential)
except:
    print("full model double exponential fit failed")
    popt_double_exponential = p0_double_exponential

a, tau1, b, tau2 = popt

print("double exponential fit")
print("a={:.2f}, tau1={:.2f}, b={:.2f}, tau2={:.2f}".format(*popt_double_exponential))

p_double_exponential = lambda x: np.exp(log_double_exponential(x, *popt_double_exponential))



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
L = log_likelihood(np.log(y), np.log(p(x)))
AIC_truncated_power_law = AIC(k, L)
BIC_truncated_power_law = BIC(k, L, n)

k = len(p0_double_exponential)
L = log_likelihood(np.log(y), np.log(p_double_exponential(x)))
AIC_double_exponential = AIC(k, L)
BIC_double_exponential = BIC(k, L, n)

k = len(p0_lognormal)
L = log_likelihood(np.log(y), np.log(p_lognormal(x)))
AIC_lognormal = AIC(k, L)
BIC_lognormal = BIC(k, L, n)

k = len(p0_exponential)
L = log_likelihood(np.log(y), np.log(p_exponential(x)))
AIC_exponential = AIC(k, L)
BIC_exponential = BIC(k, L, n)

k = len(p0_stretched_exponential)
L = log_likelihood(np.log(y), np.log(p_stretched_exponential(x)))
AIC_stretched_exponential = AIC(k, L)
BIC_stretched_exponential = BIC(k, L, n)

print("AIC")
print("truncated power law: {:.1f}".format(AIC_truncated_power_law))
print("double exponential: {:.1f}".format(AIC_double_exponential))
print("lognormal: {:.1f}".format(AIC_lognormal))
print("exponential: {:.1f}".format(AIC_exponential))
print("stretched exponential: {:.1f}".format(AIC_stretched_exponential))

print("BIC")
print("truncated power law: {:.1f}".format(BIC_truncated_power_law))
print("double exponential: {:.1f}".format(BIC_double_exponential))
print("lognormal: {:.1f}".format(BIC_lognormal))
print("exponential: {:.1f}".format(BIC_exponential))
print("stretched exponential: {:.1f}".format(BIC_stretched_exponential))


