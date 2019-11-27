#!/usr/bin/env python3
from test_inference_icu import read_data, run_MCMC, write_trace, thin_trace

if __name__ == '__main__':
    in_file = 'data/2017_18_with_pop.csv'
    data = read_data(in_file)
    for i in range(10, len(data)+1, 2):
        out_file = 'results/forecast/week{}.p'.format(i)
        trace = run_MCMC(data[0:i])
        write_trace(trace, out_file)
