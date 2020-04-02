/*
#include <boost/math/distributions/negative_binomial.hpp>
  // for negative_binomial_distribution
using boost::math::negative_binomial; // typedef provides default type is double.
using  ::boost::math::pdf; // Probability mass function.
*/
#include <iostream>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <stdio.h>      /* printf */
#include <math.h>       /* lgamma */
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
namespace py = pybind11;
/*-----------------------------------------------------------
Function ICUsim:
1) approximates on intervals dt the system of equation of a SEEIIR model 
 with time varying transmission, 
2) computes, by deterministic convolution, the daily number of 
 ICU admissions
3) simulates the dataset with negative binomial noise

 takes as a input: THETA a vector of size 6 with elements;
 - beta   : transmission rate
 - pi    : the proportion of initially immune;
 - iota  : the proportion of initially infected/infectious;
 - kappa : the factor for lockdown;
 - pIC   : the probability of IC admissions given infection;
 - eta   : the overdispersion parameter.
 
 gives as output:
 - yIC   : an integer vector for the expected daily number of IC admissions;
 
 ------------------------------------------------------------- */


const double spd     = 4;          // steps per day for the system of difference equations
const double begin   = 0;          // day of the begin of the epidemic
const double end     = 90;        // day of the end of the epidemic
const double dt      = 1/spd;      // time increment
const double N       = 66435550;   // population size
const double mean_latent_period = 4;		// in days
const double mean_infectious_period = 4.6;
const double sgm     = 2/mean_latent_period;          // rate of moving E1->E2; E2->I1
const double gmm     = 2/mean_infectious_period;       // rate of moving I1->I2; I2->r

// probability of v days elapsing between infection and IC admissions   
const std::array fEtoIC = {
		0.811443284, 0.151835341, 0.025467901,  0.006698405, 0.002425462, 0.001061865, 
		0.000524648, 0.000282547, 0.000162348,  0.000098200
};
const size_t sizemat=((end-begin)*spd)+1;  // epidemic model matrix size
const size_t lobs   = (sizemat/spd);   // length of the final observations (days of the epidemics)

std::array<double, lobs> Modelsim(py::array_t<double> THETA, std::vector<double> zetat){
  // set fixed elements:
  const auto r = THETA.unchecked<1>();
 
  // output array
  std::array<double, lobs> yIC;
  
  // parameters
  const double beta  = r(0);
  const double pi    = r(1);
  const double iota  = r(2);
  const double kappa = r(3);
  const double pIC   = r(4);
  const double eta   = r(5);
 
    // set initial values of the system of difference equations:
    Eigen::MatrixXd mat(sizemat, 7);
    double time=begin;
    mat(0,0)=time;
    mat(0,1)=(1-pi-iota)*N; 
    mat(0,2)=(iota/4)*N;
    mat(0,3)=(iota/4)*N;
    mat(0,4)=(iota/4)*N;
    mat(0,5)=(iota/4)*N;
    mat(0,6)=pi*N; 
    for (size_t t=1;t<sizemat;t++){
      // boolean that defines when to apply the lockdown effect
      // when 0 there is no effect, when 1 there is the multiplier (1+k)
      double boolK =0 ;
      if (time >= 34) {
        boolK=1;
      }
	  double infected = dt*beta*(boolK*kappa+1)*mat(t-1,1)*((mat(t-1,4)+mat(t-1,5))/N);
	  if (infected > mat(t-1,1)) {
	 	infected = mat(t-1,1);
	  }
      mat(t,1) = mat(t-1,1) - infected;
      mat(t,2) = mat(t-1,2) + infected - dt*sgm*mat(t-1,2);
      mat(t,3) = mat(t-1,3) + dt*sgm*mat(t-1,2) - dt*sgm*mat(t-1,3);
      mat(t,4) = mat(t-1,4) + dt*sgm*mat(t-1,3) - dt*gmm*mat(t-1,4);
      mat(t,5) = mat(t-1,5) + dt*gmm*mat(t-1,4) - dt*gmm*mat(t-1,5);
      mat(t,6) = mat(t-1,6) + dt*gmm*mat(t-1,6);
      time = begin + (t)*dt;
      mat(t,0) = time;
    }
    
    // number of new infections by day
    std::array<double, lobs> NNI;
    for (size_t day=0; day<(lobs); day++){
      NNI.at(day) = mat(day*spd,1) - mat((day+1)*spd,1);
    }
    
    // number of new detected IC admissions obtained via convolution
    std::array<double, lobs> NIC;
    for (size_t s = 0; s<(lobs); s++){
      double cumIC = 0;
      const size_t days_to_convolve = std::min(s, fEtoIC.size());
      for (size_t r = 0; r <= days_to_convolve; r++){
        cumIC += NNI.at(s-r) * fEtoIC.at(r);
      }
      NIC.at(s) = cumIC*zetat.at(s)*pIC;
      double sizeICs = NIC.at(s) / (eta-1); 
      yIC.at(s) = sizeICs; //R::rnbinom(sizeICs, 1/(eta) );
    }
  
  
  return yIC;
}
PYBIND11_PLUGIN(icu) {
    pybind11::module m("icu", "auto-compiled c++ extension");
    m.def("Modelsim", &Modelsim, py::return_value_policy::take_ownership);
    return m.ptr();
}
