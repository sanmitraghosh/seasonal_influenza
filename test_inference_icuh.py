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
from samplers.adaptiveKameleon import KameleonMetropolis
from samplers.kernel import GaussianKernel
from samplers.compwise_adapt_mcmc import CompWiseMCMC
from samplers.rwm_mcmc import RandomWalkMCMC
from samplers.block_mcmc import MCMC
from util.diagnostics import effective_sample_size

Transform = True
Hospitalisation = True
iterations = 50000
comp_iterations = 5000
burnin = 20000
thin = 10

Data = np.loadtxt('./fake_data/ICUsimData.txt', dtype=int)

logP = LogPosterior(Data, Transform, Hospitalisation)
start = [0.16, 0.56, 0.0000081, .1, 0.00078, 10, 0.00078, 10]
X0 = np.hstack(start)
X0 = logP._transform_from_constraint(X0)
Init_scale = 1*np.abs(X0)
cov = None
# Now run the AMGS sampler
sampler = AdaptiveMetropolis(logP, mean_est=X0, cov_est=cov, tune_interval = 1)
MCMCsampler = MCMC(sampler, logP, X0, iterations)
trace = MCMCsampler.run()[0]

#param_filename = './results/icuh_amgs.p'
#pickle.dump(trace, open(param_filename, 'wb'))

trace_post_burn = trace[burnin:,:]
py_thinned_trace_amgs = trace_post_burn[::thin,:]
amgs_ess = effective_sample_size(py_thinned_trace_amgs)

"""
# Now run the rwm sapler, with covariance initialised by compwise run
MCMCsampler = CompWiseMCMC(logP, X0, Init_scale, comp_iterations)
comptrace = MCMCsampler.run()
cov = np.cov(comptrace[2].T)
X0 = comptrace[2].mean(axis=0)
MCMCsampler = RandomWalkMCMC(logP, cov, X0, iterations)
rwmtrace = MCMCsampler.run()[0]

#param_filename = './results/icuh_rwm.p'
#pickle.dump(rwmtrace, open(param_filename, 'wb'))
rwmtrace_post_burn = rwmtrace[burnin:,:]
py_thinned_trace_rwm = rwmtrace_post_burn[::thin,:]
rwm_ess = effective_sample_size(rwmtrace_post_burn)
"""

sampler_kernel = GaussianKernel(sigma=1.5)
sampler = KameleonMetropolis(logP, sampler_kernel, X0)
MCMCsampler = MCMC(sampler, logP, X0, iterations) 
rwmtrace = MCMCsampler.run()[0]
rwmtrace_post_burn = rwmtrace[burnin:,:]
py_thinned_trace_rwm = rwmtrace_post_burn[::thin,:]
rwm_ess = effective_sample_size(rwmtrace_post_burn)

print('ESS AMGS: ', amgs_ess)
print('ESS RWM: ', rwm_ess)
sns.set_context("paper", font_scale=1)
sns.set(rc={"figure.figsize":(8,6),"font.size":10,"axes.titlesize":20,"axes.labelsize":17,
           "xtick.labelsize":6, "ytick.labelsize":6},style="white")
param_names = [r"$\beta$", r"$\pi$", r"$\iota$", r"$\kappa$", r"$p_{IC}$", r"$\eta_{IC}$", r"$p_{H}$", r"$\eta_{H}$"]
T_lines = [0.53, 0.30, 0.00001, 0.4, 0.07, 15, 0.0008, 25]
for i, p in enumerate(param_names):
        
        # Add histogram subplot
        plt.subplot(8, 2, 1 + 2 * i)
        plt.ylabel('Frequency')
        sns.kdeplot(py_thinned_trace_amgs[:, i], color='lightseagreen', legend=True, label='AMGS')
        sns.kdeplot(py_thinned_trace_rwm[:, i], color='blue', legend=True, label='RWM')
        plt.axvline(T_lines[i], linewidth=2.5, color='black')
        if i==0:
                plt.legend()

        # Add trace subplot
        plt.subplot(8, 2, 2 + 2 * i)
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
