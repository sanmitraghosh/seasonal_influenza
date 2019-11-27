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
import sys

from model.logPDFIcu import LogPosterior
from samplers.adaptiveMetropolis import AdaptiveMetropolis
from samplers.compwise_adapt_mcmc import CompWiseMCMC
from samplers.block_mcmc import MCMC
from util.diagnostics import effective_sample_size

Transform = True
Hospitalisation = False
iterations = 100000
burnin = 40000
thin = 30

def get_output_filename():
    try:
        param_filename = sys.argv[1]
    except IndexError:
        param_filename = './results/icu_amgs.p'
    return param_filename

def read_data(in_file):
    return np.loadtxt(in_file, dtype=int, delimiter=',')

def run_MCMC(data):
    logP = LogPosterior(data[:,0], Transform, Hospitalisation, catchment_pop=data[:,1])
    start = [0.16, 0.56, 0.0000081, -0.001, 0.00078, 10]
    X0 = np.hstack(start)
    Init_scale = np.array([0.05,0.03,0.000000051, 0.03, 0.0000015,0.5, 0.0000015, 0.5])

    # Now run the AMGS sampler
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
    return trace


def write_trace(trace, filename):
    pickle.dump(trace, open(filename, 'wb'))


def thin_trace(trace):
    trace_post_burn = trace[burnin:,:]
    py_thinned_trace_amgs = trace_post_burn[::thin,:]
    return py_thinned_trace_amgs


if __name__ == '__main__':
    in_file = 'data/2017_18_with_pop.csv'
    data = read_data(in_file)
    trace = run_MCMC(data)
    write_trace(trace, get_output_filename())
    thinned_trace = thin_trace(trace)
    ess = np.mean(effective_sample_size(thinned_trace))
    print('ESS AMGS: ', ess)
