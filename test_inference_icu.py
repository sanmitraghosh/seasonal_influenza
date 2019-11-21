#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import math
import pandas as pd
import seaborn as sns
import scipy.stats as stats

from model.logPDFIcu import LogPosterior
from samplers.adaptiveMetropolis import AdaptiveMetropolis
from samplers.compwise_adapt_mcmc import CompWiseMCMC
from samplers.rwm_mcmc import RandomWalkMCMC
from samplers.block_mcmc import MCMC
from util.diagnostics import effective_sample_size

Transform = True
Hospitalisation = False
iterations = 100000
comp_iterations = 10000
burnin = 40000
thin = 30

Data = np.loadtxt('data/2017_18_with_pop.csv', dtype=int, delimiter=',')

logP = LogPosterior(Data[:,0], Transform, Hospitalisation, catchment_pop=Data[:,1])
start = [0.16, 0.56, 0.0000081, -0.001, 0.00078, 10]
X0 = np.hstack(start)
Init_scale = np.array([0.05,0.03,0.000000051, 0.03, 0.0000015,0.5, 0.0000015, 0.5])

# Now run the AMGS sampler
start_amgs = time.time()
if Transform:
        X0 = logP._transform_from_constraint(X0)
        Init_scale = 1*np.abs(X0)
        cov = np.diag(Init_scale)
        sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=cov, tune_interval = 1)
        MCMCsampler = MCMC(sampler, logP, X0, iterations)
        trace = MCMCsampler.run()[0]
else:
        sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=None, tune_interval = 1)
        MCMCsampler = MCMC(sampler, logP, X0, iterations) 
        trace = MCMCsampler.run()       


end_amgs = time.time()

param_filename = './results/icu_amgs.p'
pickle.dump(trace, open(param_filename, 'wb'))



trace_post_burn = trace[burnin:,:]
py_thinned_trace_amgs = trace_post_burn[::thin,:]
amgs_ess = effective_sample_size(py_thinned_trace_amgs)


# Now run the rwm sapler, with covariance initialised by compwise run
MCMCsampler = CompWiseMCMC(logP, X0, Init_scale, comp_iterations)
start_rwm = time.time()
comptrace = MCMCsampler.run()
if  Transform:
        cov = np.cov(comptrace[2].T)
        X0 = comptrace[2].mean(axis=0)
        MCMCsampler = RandomWalkMCMC(logP, cov, X0, iterations)
        rwmtrace = MCMCsampler.run()[0]
else:
        cov = np.cov(comptrace[0].T)
        X0 = comptrace[0].mean(axis=0)
        MCMCsampler = RandomWalkMCMC(logP, cov, X0, iterations)
        rwmtrace = MCMCsampler.run()
end_rwm = time.time()


param_filename = './results/icu_rwm.p'
pickle.dump(rwmtrace, open(param_filename, 'wb'))
rwmtrace_post_burn = rwmtrace[burnin:,:]


py_thinned_trace_rwm = rwmtrace_post_burn[::thin,:]
rwm_ess = effective_sample_size(rwmtrace_post_burn)
print('ESS AMGS: ', amgs_ess)
print('ESS RWM: ', rwm_ess)
print('AMGS time taken: ', end_amgs - start_amgs)
print('RWM time taken: ', end_rwm - start_rwm)
sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(8,6),"font.size":10,"axes.titlesize":20,"axes.labelsize":17,
           "xtick.labelsize":6, "ytick.labelsize":6},style="white")
param_names = [r"$\beta$", r"$\pi$", r"$\iota$", r"$\kappa$", r"$pIC$", r"$\eta_{IC}$"]
T_lines = [0.56, 0.36, 0.0000051, -0.35, 0.00015,15]
for i, p in enumerate(param_names):
        
        # Add histogram subplot
        plt.subplot(6, 2, 1 + 2 * i)
        plt.ylabel('Frequency')
        sns.kdeplot(py_thinned_trace_amgs[:, i], color='lightseagreen', legend=True, label='AMGS')
        sns.kdeplot(py_thinned_trace_rwm[:, i], color='blue', legend=True, label='RWM')
        plt.axvline(T_lines[i], linewidth=2.5, color='black')
        if i==0:
                plt.legend()

        # Add trace subplot
        plt.subplot(6, 2, 2 + 2 * i)
        plt.ylabel(p, fontsize=20)  
        plt.plot(py_thinned_trace_amgs[:, i], alpha=0.5, color='lightseagreen')
        plt.plot(py_thinned_trace_rwm[:, i], alpha=0.5, color='blue')
        

plt.show()

pair_pd = pd.DataFrame(data=py_thinned_trace_rwm, index=range(len(py_thinned_trace_rwm )), columns=param_names)
pair_pd.head()


g = sns.PairGrid(pair_pd)
g.map_diag(sns.distplot, bins=30, kde=False, hist_kws={"histtype":'bar',"rwidth":0.8,"linewidth": 0.5,
                          "alpha": 0.5, "color": "b"})

g.map_offdiag(plt.scatter, color='seagreen', s=10, alpha=0.1, edgecolors='none')#linewidth=0.1, marker="o"
plt.subplots_adjust(bottom=0.065)
for ax in g.axes[:,0]:
    ax.get_yaxis().set_label_coords(-0.265,0.5)

plt.show()
