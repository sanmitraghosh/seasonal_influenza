import matplotlib.pyplot as plt
from model.logPDFIcu import LogPosterior
import numpy as np
import pandas as pd
import pickle
import random
import seaborn as sns
from util.diagnostics import effective_sample_size

burnin = 40000
thin = 30
param_filename = './results/icu_amgs.p'

trace = pickle.load(open(param_filename, 'rb'))
trace_post_burn = trace[burnin:,:]
py_thinned_trace_amgs = trace_post_burn[::thin,:]
amgs_ess = effective_sample_size(py_thinned_trace_amgs)


print('ESS AMGS: ', amgs_ess)
sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(8,6),"font.size":10,"axes.titlesize":20,"axes.labelsize":17,
           "xtick.labelsize":6, "ytick.labelsize":6},style="white")
param_names = [r"$\beta$", r"$\pi$", r"$\iota$", r"$\kappa$", r"$pIC$", r"$\eta_{IC}$"]

ppc_samples = 1000
new_values_amgs = []
Data = np.loadtxt('data/2017_18_with_pop.csv', dtype=int, delimiter=',')
true_values = Data[:,0]
weeks = len(true_values)
Transform = True
Hospitalisation = False
logP = LogPosterior(true_values, Transform, Hospitalisation, catchment_pop=Data[:,1])
for ind in random.sample(range(0, np.size(py_thinned_trace_amgs, axis=0)), ppc_samples): 
    ppc_sol = logP.posterior_draws(py_thinned_trace_amgs[ind, :])
    new_values_amgs.append(ppc_sol.copy())

new_values_amgs = np.array(new_values_amgs)
median_values_amgs = np.percentile(new_values_amgs,q=50,axis=0)
mean_values_amgs = np.mean(new_values_amgs,axis=0)
CriL_ppc_amgs = np.percentile(new_values_amgs,q=2.5,axis=0)
CriU_ppc_amgs = np.percentile(new_values_amgs,q=97.5,axis=0)
time = np.arange(1, weeks+1)
plt.plot(time, true_values,'o--',ms=10, color='orange', label='ICU admissions 2017/18')
plt.plot(time, median_values_amgs, color='green', label='Median_AMGS')
plt.plot(time, mean_values_amgs, color='grey', label='Mean_AMGS')
plt.plot(time, CriL_ppc_amgs,'-', color='blue', label='lQuant_AMGS')
plt.plot(time, CriU_ppc_amgs,'-', color='blue', label='UQuant_AMGS')
for i in [19,28,75,92,131,140,179,196]:
    plt.axvline(i/7, color='red')
plt.ylabel(r"$Y_{ICU}$")
plt.xlabel('weeks')
plt.xlim((1, weeks))
plt.legend()
plt.show()
