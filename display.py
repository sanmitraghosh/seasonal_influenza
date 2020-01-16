import matplotlib.pyplot as plt
from model.logPDFIcu import LogPosterior
import numpy as np
import pandas as pd
import pickle
import random
import seaborn as sns
from util.diagnostics import effective_sample_size

PARAM_NAMES = [r"$\beta$", r"$\pi$", r"$\iota$", r"$\kappa$", r"$pIC$", r"$\eta_{IC}$"]

def init(args):
    trace = pickle.load(open(args.result_file, 'rb'))
    trace_post_burn = trace[args.burnin:,:]
    py_thinned_trace_amgs = trace_post_burn[::args.thin,:]
    return py_thinned_trace_amgs


def ESS(trace, args):
    amgs_ess = effective_sample_size(trace)
    print('ESS AMGS: ', amgs_ess)

def predict(trace, args):
    sns.set_context("paper", font_scale=1)
    sns.set(rc={"figure.figsize":(8,6),"font.size":10,"axes.titlesize":20,"axes.labelsize":17,
               "xtick.labelsize":6, "ytick.labelsize":6},style="white")

    ppc_samples = 1000
    new_values_amgs = []
    Data = np.loadtxt('data/2018_19_with_pop.csv', dtype=int, delimiter=',')
    true_values = Data[:,0]
    weeks = len(true_values)
    Transform = True
    Hospitalisation = False
    logP = LogPosterior(true_values, Transform, Hospitalisation, catchment_pop=Data[:,1])
    output = []
    #for ind in random.sample(range(0, np.size(trace, axis=0)), ppc_samples): 
    for ind in range(trace.shape[0]):
        ppc_sol = logP.posterior_draws(trace[ind, :])
        new_values_amgs.append(ppc_sol.copy())
        if args.include_params:
            out = list(trace[ind,:]) + list(new_values_amgs[-1])
            output.append(out)
        else:
            output.append(new_values_amgs[-1])

    if args.output is not None:
        pickle.dump(output, open(args.output + '.p', 'wb'))

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
    return plt

def params(trace, args):
    for i, p in enumerate(PARAM_NAMES):
        # Add histogram subplot
        plt.subplot(6, 2, 1 + 2 * i)
        plt.ylabel('Frequency')
        sns.kdeplot(trace[:, i], color='lightseagreen', legend=True, label='AMGS')
        if i==0:
                plt.legend()

        # Add trace subplot
        plt.subplot(6, 2, 2 + 2 * i)
        plt.ylabel(p, fontsize=20)  
        plt.plot(trace[:, i], alpha=0.5, color='lightseagreen')
    return plt

def correlations(trace, args):
    pair_pd = pd.DataFrame(data=trace, index=range(len(trace)), columns=PARAM_NAMES)
    pair_pd.head()

    g = sns.PairGrid(pair_pd)
    g.map_diag(sns.distplot, bins=30, kde=False,
               hist_kws={"histtype":'bar',"rwidth":0.8,"linewidth": 0.5,
                         "alpha": 0.5, "color": "b"})

    g.map_offdiag(plt.scatter, color='seagreen', s=10, alpha=0.1, edgecolors='none')
    plt.subplots_adjust(bottom=0.065)
    for ax in g.axes[:,0]:
        ax.get_yaxis().set_label_coords(-0.265,0.5)

    return plt

OPTIONS = {
    'ESS': ESS,
    'predict': predict,
    'params': params,
    'correl': correlations,
}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('display', help='What figure or statistic you want to see',
                       choices=OPTIONS.keys())
    parser.add_argument('-r', '--result-file', default='results/icu_amgs.p',
                        help='The file holding the results (pickled) to use')
    parser.add_argument('-b', '--burnin', type=int, default=40_000,
                        help='How many iterations to discard as burnin')
    parser.add_argument('-t', '--thin', type=int, default=30,
                        help='How to thin the chain')
    parser.add_argument('-o', '--output', default=None,
                        help='File to output to (otherwise shown on screen). Ignored for ESS.')
    parser.add_argument('-p', '--include-params', action='store_true',
                        help='For predict only: if the paramater values used to in each '
                             'simulation should be included in the pickle output.')
    args = parser.parse_args()
    trace = init(args)
    plot = OPTIONS[args.display](trace, args)
    if plot is not None:
        if args.output is None:
            plt.show()
        else:
            plt.savefig(args.output)
