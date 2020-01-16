#!/usr/bin/env python3
from test_inference_icu import read_data, run_MCMC, write_trace, thin_trace

if __name__ == '__main__':
    in_file = 'data/2017_18_with_pop.csv'
    data = read_data(in_file)
    for i in range(1, 34):
        print('For week {}'.format(i))
        data_to_use = (data[0][0:i], data[1])
        assert len(data_to_use[0]) == i
        out_file = 'results/forecast/week{}.p'.format(i)
        trace = run_MCMC(data_to_use)
        write_trace(trace, out_file)
