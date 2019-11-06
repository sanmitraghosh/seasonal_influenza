from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
import scipy.stats as stats
import pickle


class CompWiseMCMC(object):
    def __init__(self, target, x0, scale, iterations=150000):

        self._target = target
        self._iterations = iterations
        self._x0 = x0
        self._sub_blocks = target.n_params
        self._scale = scale

        self._verbose = True

    def _run_independent(self):
        # Report the current settings
        if self._verbose:
            print('Running adaptive Independent MCMC')
            print('Total number of iterations: ' + str(self._iterations))

    
        # Problem dimension
        d = self._target.n_params

        # Initial starting parameters
        current = self._x0

        # Chain of stored samples

        chain = np.zeros((self._iterations,d))
        if self._target._transform:
            tx_chain = np.zeros((self._iterations,d))

        # Initial acceptance rate 
        acceptance = np.zeros(d)
       

        for i in range(self._iterations):
            
            for j in range(self._sub_blocks):

                # Copy the current whole parameter vector
                proposed = current.copy()

                # Sequentially update (locally) the copied version, which is the proposed

                proposed[j] = stats.norm(current[j], self._scale[j]).rvs()

                # Evaluate the logP if first iteration of the chain
                if i==0 and j==0:
                    current_log_target = self._target(current)

                # Evaluate the logP of the proposed
                proposed_log_target = self._target(proposed)

                # Routine MH step
                log_ratio = proposed_log_target - current_log_target 
                log_ratio = min(np.log(1), log_ratio)

                accepted = 0
                if np.isfinite(proposed_log_target):
                    if log_ratio > np.log(np.random.rand(1)):
                        accepted = 1
                        current = proposed
                        current_log_target = proposed_log_target  

                # Free the current's copy
                proposed = None     
                
                # Update acceptance rates
                acceptance[j] = acc_rate = (i * acceptance[j] + float(accepted)) / (i + 1)      

                # Adapt proposal scale
                if i > 100 and i % 10 == 0:

                    if acc_rate <= 0.2:
                        self._scale[j] *= 0.9
                    elif acc_rate >= 0.3:
                        self._scale[j] *= 1.1  
                          
            # Store the current
            if self._target._transform:
                chain[i,:] = self._target._transform_to_constraint(current)
                tx_chain[i,:] = current
            else:
                chain[i,:] = current

             

            # Print-outs
            if self._verbose and ((i % 500 == 0) or (i==self._iterations-1)):
                print('Iteration ' + str(i) + ' of ' + str(self._iterations))
                print('  Average Acceptance rate: ' + str(acceptance.mean()))

        if self._target._transform:
            return chain, self._scale, tx_chain
        else:
            return chain, self._scale
    
    def run(self):

        return self._run_independent()

