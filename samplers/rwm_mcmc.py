from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import math

class RandomWalkMCMC(object):
    
    _verbose=True
    _adapt=True
    def __init__(self, target, covariance, x0, iterations=150000):

        self._target = target
        # Total number of iterations
        self._iterations = iterations # adjust according to chpt

        self._cov_est = covariance   # if chkpt then load chpts of sampler
        
        self._x0 = x0 # adjust according to chpt
        self._acceptance_target = None
        self._adaptations = math.floor(iterations/2.0)

    def run(self):
        # Report the current settings
        if self._verbose:
            print('Running RWM Blocked MCMC')
            print('Total number of iterations: ' + str(self._iterations))

    
        # Problem dimension
        d = self._target.n_params

        # Initial starting parameters
        current = self._x0 

        # Chain of stored samples

        chain = np.zeros((self._iterations,d)) 
        if self._target._transform:
            tx_chain = np.zeros((self._iterations,d))
        # Initial acceptance rate (value doesn't matter)

        acceptance = 0 

        for i in range(self._iterations):

            proposed = np.random.multivariate_normal(current, self._cov_est)

            if i==0:
                current_log_target = self._target(current)

            proposed_log_target = self._target(proposed)

            log_ratio = (proposed_log_target - current_log_target) #+ (np.sum(np.log(proposed)) + np.sum(np.log(current)) )

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
            else:
                chain[i,:] = current

            # Update acceptance rate
            acceptance = (i * acceptance + float(accepted)) / (i + 1)

            # Adapt proposal covariance
            if i < self._adaptations and i % 50 == 0 and self._adapt:

                if acceptance <= 0.2:
                    self._cov_est *= 0.9
                elif acceptance >= 0.3:
                    self._cov_est *= 1.1                

            # Report
            if self._verbose and i % 2000 == 0:
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))

                print('  Acceptance rate: ' + str(acceptance))

        # Return generated chain
        if self._target._transform:
            return chain, tx_chain
        else:
            return chain

        
