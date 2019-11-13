import numpy as np
import scipy
import scipy.stats as stats
from CPP import icu
from CPP import icuh


class LogPosterior(object):
    _fixed_length = 33
    def __init__(self, data, transform=False, hospital=False, realdata = False, catchment_pop = None, info_prior=True):
        
        self._sim_length = int(len(data)/2)
        if hospital:
            self._icu_data = data[self._sim_length:]
            self._hosp_data = data[:self._sim_length]
            self._simulator = icuh
            self.n_params = 8
        else:
            if len(data)>self._fixed_length:
                self._icu_data = data[self._sim_length:]
            else:
                self._icu_data = data

            self._hosp_data = None           
            self._simulator = icu
            self.n_params = 6
        self._hospital = hospital
        self._realdata = realdata
        self._population = 55268100
        self._transform = transform
        self._info_prior = info_prior

        ### Currently ignore this bit as detection probs are hardcoded in C++
        if catchment_pop is not None:
            catchment_pop = np.array(catchment_pop,dtype=float)
            self._icu_pop = catchment_pop[self._sim_length:]/self._population \
                            if len(data)>self._fixed_length else catchment_pop/self._population
            if hospital:
                self._hosp_pop = catchment_pop[:self._sim_length] / self._population
                
    def icu_and_hosp_lik(self, _param, simulate=False) :

        model_sim = np.array(self._simulator.Modelsim(_param))
        ll = 0.0
        if not(self._hospital):
            icu_sim = model_sim
            _eta = _param[-1]
            
            for i in range(len(self._icu_data)):
                if icu_sim[i]==0 and self._icu_data[i] != 0:
                    ll += -np.inf
                elif icu_sim[i]==0 and self._icu_data[i] == 0:
                    ll +=  0.
                else:
                    ll += stats.nbinom.logpmf(self._icu_data[i],n=icu_sim[i],p=1/_eta)
        else:
            icu_sim = model_sim[self._fixed_length:self._fixed_length+self._sim_length]
            hosp_sim = model_sim[:self._sim_length]
            """ 
            if self._realdata:
                icu_sim *= self._icu_pop
                hosp_sim *= self._hosp_pop
            """
            _etaH, _etaIC = _param[7], _param[5]

            if simulate:

                icu_draws = stats.nbinom(n=icu_sim,p=1/_etaIC).rvs()
                hosp_draws = stats.nbinom(n=hosp_sim,p=1/_etaH).rvs()
                return np.append(hosp_draws,icu_draws)


            for i in range(len(self._icu_data)):
                if icu_sim[i]==0 and self._icu_data[i] != 0:
                    ll += -np.inf
                elif icu_sim[i]==0 and self._icu_data[i] == 0:
                    ll +=  0.
                else:
                    ll += stats.nbinom.logpmf(self._icu_data[i],n=icu_sim[i],p=1/_etaIC)
            for i in range(len(self._hosp_data)):
                if hosp_sim[i]==0 and self._hosp_data[i] != 0:
                    ll += -np.inf
                elif hosp_sim[i]==0 and self._hosp_data[i] == 0:
                    ll +=  0.
                else:
                    ll += stats.nbinom.logpmf(self._hosp_data[i],n=hosp_sim[i],p=1/_etaH)
        
        return ll 

    def posterior_draws(self, params):
        return self.icu_and_hosp_lik(params, True)

    
    def _transform_to_constraint(self, transformed_parameters):

        Tx_THETA = transformed_parameters
        Utx_beta  = np.exp(Tx_THETA[0])
        Utx_pi    = np.exp(Tx_THETA[1])/(np.exp(Tx_THETA[1])+1)
        Utx_iota  = np.exp(Tx_THETA[2])/((np.exp(Tx_THETA[1])+1)*(np.exp(Tx_THETA[2])+1))
        Utx_kappa = np.exp(Tx_THETA[3]) - 1
        Utx_pIC   = np.exp(Tx_THETA[4])/(np.exp(Tx_THETA[4])+1)
        Utx_etaIC   = np.exp(Tx_THETA[5] + 1.)
        if self._hospital:
            Utx_pH   = np.exp(Tx_THETA[6])/(np.exp(Tx_THETA[6])+1)
            Utx_etaH   = np.exp(Tx_THETA[7] + 1.)
            return np.array([Utx_beta, Utx_pi, Utx_iota, Utx_kappa, Utx_pIC, Utx_etaIC, Utx_pH, Utx_etaH])
        else:
            return np.array([Utx_beta, Utx_pi, Utx_iota, Utx_kappa, Utx_pIC, Utx_etaIC])

    def _transform_from_constraint(self, untransformed_parameters):
        
        Utx_THETA = untransformed_parameters
        tx_beta  = np.log(Utx_THETA[0])  
        tx_pi    = np.log( Utx_THETA[1]/(1 - Utx_THETA[1]) )
        tx_iota  = np.log( Utx_THETA[2]/(1 - (Utx_THETA[1] + Utx_THETA[2]) ) )
        tx_kappa = np.log(Utx_THETA[3]+1)
        tx_pIC   = np.log(Utx_THETA[4]/(1 - Utx_THETA[4]))
        tx_etaIC   = np.log(Utx_THETA[5] - 1.)  #STAN
        if self._hospital:
            tx_pH   = np.log(Utx_THETA[6]/(1 - Utx_THETA[6]))
            tx_etaH   = np.log(Utx_THETA[7] - 1.)  #STAN
            return np.array([tx_beta, tx_pi, tx_iota, tx_kappa, tx_pIC, tx_etaIC, tx_pH, tx_etaH])
        else:
            return np.array([tx_beta, tx_pi, tx_iota, tx_kappa, tx_pIC, tx_etaIC])

    def __call__(self, parameters):
        
        if self._transform:
            _Tx_THETA = parameters.copy()
            THETA = self._transform_to_constraint(_Tx_THETA)
            _Tx_beta  = _Tx_THETA[0]
            _Tx_pi    = _Tx_THETA[1]
            _Tx_iota  = _Tx_THETA[2]
            _Tx_kappa = _Tx_THETA[3]
            _Tx_pIC   = _Tx_THETA[4]
            _Tx_etaIC   = _Tx_THETA[5]     
            if self._hospital:
                _Tx_pH   = _Tx_THETA[6]
                _Tx_etaH = _Tx_THETA[7]
        else:
            THETA = parameters.copy()

        beta  = THETA[0]
        pi    = THETA[1]
        iota  = THETA[2]
        kappa = THETA[3]
        pIC   = THETA[4]
        etaIC = THETA[5]     
        if self._hospital:
            pH = THETA[6]
            etaH  = THETA[7]   

        # Get the ICU/H likelihood
        
        if ( (pi+iota>=1.) or (beta<=0.) or (kappa<=-1.) or \
            (pIC>=1.) or (pIC<=0.) or (etaIC<=1.) or \
            (self._hospital and pH>=1.) or (self._hospital and pH<=0.) or \
            (self._hospital and etaH<=1.)):

            log_likelihood = np.inf
        else:
            log_likelihood = self.icu_and_hosp_lik(THETA)
        
        # Now calculate the necessary priors for ODE model params

        if self._transform:
            if self._info_prior:
                #

                logPrior_beta = stats.halfnorm(0,1).logpdf(beta) + _Tx_beta
                logPrior_pi = stats.beta.logpdf(pi,37.5,62.5) + \
                                stats.uniform.logpdf(iota,0.,1e-1) +\
                                ((_Tx_pi)-2*np.log((1+np.exp(_Tx_pi)))) +\
                                ((_Tx_iota)-(2*np.log(1+np.exp(_Tx_iota))+np.log(1+np.exp(_Tx_pi)))) 

                logPrior_iota = stats.uniform.logpdf(iota,0.,1e-1) +\
                                ((_Tx_iota)-(2*np.log(1+np.exp(_Tx_iota))+np.log(1+np.exp(_Tx_pi)))) 
                logPrior_kappa = stats.norm.logpdf(_Tx_kappa, 0, 1) 
                logPrior_pIC = stats.beta(2,1).logpdf(pIC) +\
                                ((_Tx_pIC)-(2*np.log(1+np.exp(_Tx_pIC))))
                logPrior_etaIC = stats.gamma.logpdf(etaIC,2,scale=10) + _Tx_etaIC 
                if self._hospital:
                    logPrior_pH = stats.beta(2,1).logpdf(pH) +\
                                ((_Tx_pH)-(2*np.log(1+np.exp(_Tx_pH))))
                    logPrior_etaH = stats.gamma.logpdf(etaH,2,scale=10) + _Tx_etaH  
            else:
                logPrior_beta = stats.uniform.logpdf(beta,0.,4.) + _Tx_beta
                logPrior_pi = stats.beta.logpdf(pi,37.5,62.5) + \
                                stats.uniform.logpdf(iota,0.,1e-1) +\
                                ((_Tx_pi)-2*np.log((1+np.exp(_Tx_pi)))) +\
                                ((_Tx_iota)-(2*np.log(1+np.exp(_Tx_iota))+np.log(1+np.exp(_Tx_pi)))) 

                logPrior_iota = stats.uniform.logpdf(iota,0.,1e-1) +\
                                ((_Tx_iota)-(2*np.log(1+np.exp(_Tx_iota))+np.log(1+np.exp(_Tx_pi)))) 
                logPrior_kappa = stats.norm.logpdf(_Tx_kappa, 0, 1) 
                logPrior_pIC = stats.uniform.logpdf(pIC, 0., 1.) +\
                                ((_Tx_pIC)-(2*np.log(1+np.exp(_Tx_pIC))))
                logPrior_etaIC = stats.uniform.logpdf(etaIC, 1.0, 100.0) + _Tx_etaIC 
                if self._hospital:
                    logPrior_pH = stats.uniform.logpdf(pH, 0., 1.) +\
                                ((_Tx_pH)-(2*np.log(1+np.exp(_Tx_pH))))
                    logPrior_etaH = stats.uniform.logpdf(etaH, 1.0, 100.0) + _Tx_etaH                   
        else:

            logPrior_beta = stats.uniform.logpdf(beta,0.,4.)
            logPrior_pi = stats.beta.logpdf(pi,37.5,62.5)   
            logPrior_iota = stats.uniform.logpdf(iota, 0., 1e-1)
            logPrior_kappa = stats.lognorm.logpdf(kappa+1, s=1., scale=1.) 
            logPrior_pIC = stats.uniform.logpdf(pIC, 0., 1.)
            logPrior_etaIC = stats.uniform.logpdf(etaIC, 1.0, 100.0)
            if self._hospital:
                logPrior_pH = stats.uniform.logpdf(pH, 0., 1.)
                logPrior_etaH = stats.uniform.logpdf(etaH, 1.0, 100.0)
        if self._hospital:
            log_prior = logPrior_beta + logPrior_pi + logPrior_iota + logPrior_kappa + \
                        logPrior_pIC + logPrior_etaIC + logPrior_pH + logPrior_etaH 
        else:
            log_prior = logPrior_beta + logPrior_pi + logPrior_iota + logPrior_kappa + \
                        logPrior_pIC + logPrior_etaIC

        return log_likelihood + log_prior
                   
    def n_parameters(self):
        return self.n_params
    def log_pdf(self, x):
        return self.__call__(x)
    def grad(self, X):
        return NotImplemented
            
        


