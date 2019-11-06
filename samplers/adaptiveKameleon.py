import scipy.stats as stats
import scipy
import numpy as np


class KameleonMetropolis(object):
    '''
    Kernel adaptive Metropolis algorithm. This class implements the kernelised adaptation 
    and proposal building code
    '''
    def __init__(self, target, kernel, x0=None, stop_adapt=np.inf):

        self.method ='KM'
        self._target = target
        self.nu2 = np.log(0.1)#(log 0f 0.1)###np.log((2.38 ** 2) /self._target.n_params) works well
        self.gamma = 0.003##0.001 has worked with ICU logged 0.2 for banana
        self.Z = None
        self._kernel = kernel
        self._discard = 199
        self._num_samples_Z = 500
        self._stop_adapt = stop_adapt

        if x0 is None:
            self._x0 = 2*np.ones(self._target.n_params)
        else:
            self._x0 = x0

        self._target.n_params = self._target.n_parameters()
        self.gamma_dual = 0.05
        self.t0 = 10
        self.mu = 2.5*self.nu2#np.log(5*((2.38 ** 2) /self._target.n_params)) ## 2.5*self.nu2 works well
        self.Ht = 0.0
        self.iter = 0
        self.cov_est = None
    def compute_constants(self, current):

        if self.Z is None:
            start_cov = np.abs(self._x0)
            R = np.diag(start_cov)
            #R = self.gamma * np.eye(self._target.n_params)#np.exp(self.gamma)

        else:
            M = 2 * self._kernel.gradient(current, self.Z)
            H = self._kernel.centring_matrix(len(self.Z))
            R = self.gamma**2  * np.eye(len(current)) + np.exp(self.nu2)  * M.T.dot(H.dot(M))
            
            try:
                L = np.linalg.cholesky(R)
            except np.linalg.LinAlgError:
                # some really crude check for PSD (which only corrects for orunding errors
                R = R + 1e-6*np.eye(self._target.n_params)
 
        return current.copy(), R

    def scale_adapt(self, learning_rate, accepted_or_not):
        
        self.nu2 += learning_rate * (accepted_or_not - 0.234) 
        
        #self._momentum  =  self._beta* self._momentum + learning_rate * (accepted_or_not - 0.234) 
        #self.nu2 = self.nu2 - self._momentum
        #self._kernel.width = np.exp(np.log(self._kernel.width) + learning_rate * (np.exp(accepted_or_not) - 0.234))
        #self.nu2 = np.exp(np.log(self.nu2) + learning_rate * (np.exp(accepted_or_not) - 0.234))
        #self.gamma = np.exp(np.log(self.gamma) + learning_rate * (np.exp(accepted_or_not) - 0.234))
    def scale_dualavg_adapt(self, iteration, learning_rate, acceptance):
        
        t = (iteration - self._discard + 1.0)
        self.Ht = (1 - (1/(t + self.t0)))*self.Ht +  (1/(t + self.t0))*(0.234 - np.exp(acceptance)) 
        
        self.xt = self.mu - (np.sqrt(t)/self.gamma_dual)*self.Ht
        
        self.nu2  = (1-learning_rate) * self.nu2 + learning_rate*self.xt

    def adapt(self, iteration, samples, accepted_or_not, acceptance, chain_history):
        self.iter = iteration
        if iteration>self._discard:
            learning_rate = (iteration - self._discard + 1.0) ** -0.6
            self.scale_adapt(learning_rate,accepted_or_not)
            #self.scale_dualavg_adapt(iteration,learning_rate,acceptance)            
            if np.random.rand() < learning_rate:

                #self._kernel.width = self._kernel.get_sigma_median_heuristic(chain_history)
                if iteration < self._discard + self._num_samples_Z:
                    self.Z = chain_history[self._discard:(iteration + 1),:]
                else:
                    if iteration<self._stop_adapt:
                        #if iteration % 1500 == 0:
                        #   self._kernel.width = self._kernel.get_sigma_median_heuristic(chain_history)
                        inds = np.random.randint(iteration - self._discard, size=self._num_samples_Z) + self._discard
                        unique_inds = np.unique(inds)
                        self.Z = chain_history[unique_inds,:]
            
#        if iteration==20000:
#            self._kernel.width = self._kernel.get_sigma_median_heuristic(chain_history)

    def proposal(self, y):
            iter = self.iter
            mu, R = self.compute_constants(y)
            
            new = np.random.multivariate_normal(mu, R)
            mu_n, R_n = self.compute_constants(new)
            
            log_q_new = stats.multivariate_normal(mu, R, allow_singular=True).logpdf(new)         
            log_q_old = stats.multivariate_normal(mu_n, R_n, allow_singular=True).logpdf(mu)
            return new, log_q_old, log_q_new

    