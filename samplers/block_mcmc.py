from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np


class MCMC(object):
    
    _verbose=True
    def __init__(self, sampler, target, x0, iterations=150000):

        self._target = target
        # Total number of iterations
        self._iterations = iterations # adjust according to chpt

        self._sampler = sampler    # if chkpt then load chpts of sampler

        self._x0 = x0 # adjust according to chpt
        self._acceptance_target = None
        

    def run(self):
        # Report the current settings
        if self._verbose:
            print('Running adaptive Blocked MCMC')
            print('Total number of iterations: ' + str(self._iterations))

    
        # Problem dimension
        d = self._target.n_parameters()#n_params#

        # Initial starting parameters
        current = self._x0 

        # Chain of stored samples

        chain = np.zeros((self._iterations,d)) 
        if self._target._transform:
            tx_chain = np.zeros((self._iterations,d))

        log_pdfs = np.zeros(self._iterations) 
        if self._target._transform:
            tx_log_pdfs = np.zeros(self._iterations)            

        # Initial acceptance rate (value doesn't matter)

        acceptance = 0 

        for i in range(self._iterations):

            if self._sampler.method == 'KM':
                proposed, log_q_old, log_q_new = self._sampler.proposal(current)
            else:
                proposed = self._sampler.proposal(current)

            if i==0:
                current_log_target = self._target(current)

            proposed_log_target = self._target(proposed)
            if self._sampler.method == 'KM':
                log_ratio = proposed_log_target - current_log_target + log_q_old - log_q_new
                
            else:
                log_ratio = (proposed_log_target - current_log_target)

            log_ratio = min(np.log(1), log_ratio)
            accepted = 0
            if np.isfinite(proposed_log_target):
                if log_ratio > np.log(np.random.rand(1)):
                    accepted = 1
                    current = proposed
                    current_log_target = proposed_log_target

            proposed = None
            # Store the current in the chain
            if self._target._transform:
                chain[i,:] = self._target._transform_to_constraint(current)
                tx_chain[i,:] = current
                tx_log_pdfs[i] = current_log_target
            else:
                chain[i,:] = current
                log_pdfs[i] = current_log_target

            # Update acceptance rate
            acceptance = (i * acceptance + float(accepted)) / (i + 1)
            if self._target._transform:
                self._sampler.adapt(i, current, accepted, log_ratio, tx_chain[:i,:])
            else:

                if self._sampler.method == 'IGM':

                    self._sampler.adapt(i, current, accepted, log_ratio, chain[:i,:],log_pdfs)
                else:
                    self._sampler.adapt(i, current, accepted, log_ratio, chain[:i,:])

            # Report
            if self._verbose and i % 2000 == 0:
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))

                print('  Acceptance rate: ' + str(acceptance))
                
                
        # Return generated chain
        if self._target._transform:
            return chain, tx_chain
        else:
            return chain
