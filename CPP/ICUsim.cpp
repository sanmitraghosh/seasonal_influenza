/*
#include <boost/math/distributions/negative_binomial.hpp>
  // for negative_binomial_distribution
using boost::math::negative_binomial; // typedef provides default type is double.
using  ::boost::math::pdf; // Probability mass function.
*/
#include <iostream>
#include <vector>
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
2) computes, by deterministic convolution, the weekly number of 
 ICU admissions
3) simulates the dataset with negative binomial noise

 takes as a input: THETA a vector of size 6 with elements;
 - beta   : transmission rate
 - pi    : the proportion of initially immune;
 - iota  : the proportion of initially infected/infectious;
 - kappa : the factor for school holidays;
 - pIC   : the probability of IC admissions given infection;
 - eta   : the overdispersion parameter.
 
 gives as output:
 - yIC   : an integer vector (usually of size 33) 
 for the weekly number of IC admissions;
 
 ------------------------------------------------------------- */


std::vector<double> Modelsim(py::array_t<double> THETA, std::vector<double> zetat){
  // set fixed elements:
  auto r = THETA.unchecked<1>();
  double spd     = 4;          // steps per day for the system of difference equations
  double begin   = 0;          // day of the begin of the epidemic
  double end     = 231;        // day of the end of the epidemic
  double dt      = 1/spd;      // time increment
  double N       = 55977178;   // population size
  double sgm     = 1;          // rate of moving E1->E2; E2->I1
  double gmm     = 0.52;       // rate of moving I1->I2; I2->r

  std::vector<double> fEtoIC= {0.811443284,0.151835341,0.025467901,
                         0.006698405,0.002425462,0.001061865,
                         0.000524648,0.000282547,0.000162348,
                         0.000098200};        // probability of v weeks elapsing between 
                                              // infection and IC admissions   
  int sizemat=((end-begin)*spd)+1;  // epidemic model matrix size
  int lobs   = (sizemat/(7*spd));   // length of the final observations (weeks of the epidemics)
 
  // output vector
  std::vector<double> yIC(lobs);
  
  // parameters
  double beta  = r(0);
  double pi    = r(1);
  double iota  = r(2);
  double kappa = r(3);
  double pIC   = r(4);
  double eta   = r(5);
 
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
    for (int t=1;t<sizemat;t++){
      // boolean that defines when to apply the holiday effect
      // when 0 there is no effect, when 1 there is the multiplier (1+k)
      double boolK =0 ;
      if (((time>=19)&&(time<=28))||((time>=75)&&(time<=92))||
          ((time>=131)&&(time<=140))||((time>=179)&&(time<=196))){
        boolK=1;
      }
      mat(t,1) = mat(t-1,1) - dt*beta*(boolK*kappa+1)*mat(t-1,1)*((mat(t-1,4)+mat(t-1,5))/N);
      mat(t,2) = mat(t-1,2) + dt*beta*(boolK*kappa+1)*mat(t-1,1)*((mat(t-1,4)+mat(t-1,5))/N) - dt*sgm*mat(t-1,2);
      mat(t,3) = mat(t-1,3) + dt*sgm*mat(t-1,2) - dt*sgm*mat(t-1,3);
      mat(t,4) = mat(t-1,4) + dt*sgm*mat(t-1,3) - dt*gmm*mat(t-1,4);
      mat(t,5) = mat(t-1,5) + dt*gmm*mat(t-1,4) - dt*gmm*mat(t-1,5);
      mat(t,6) = mat(t-1,6) + dt*gmm*mat(t-1,6);
      time = begin + (t)*dt;
      mat(t,0) = time;
    }
    
    // number of new infections by week
    int thin=(7*spd);
    std::vector<double> NNI(lobs);
    for (int s=0;s<(lobs);s++){
      NNI[s] = mat(s*thin,1) - mat((s+1)*thin,1);
    }
    
    // number of new detected IC admissions obtained via convolution
    // and Negbinom
    std::vector<double> NIC(lobs);
    for (int s=0;s<(lobs);s++){
      double cumIC=0;
      int R=std::min(s, 9);
      for (int r=0; r<(R+1); r++){
        cumIC+=NNI[s-r]*fEtoIC[r];
      }
      NIC[s] = cumIC*zetat.at(s)*pIC;
      // obsevrations
      double sizeICs = NIC[s]/(eta-1); 
      yIC[s] = sizeICs; //R::rnbinom(sizeICs, 1/(eta) );
    }
  
  
  return yIC;
}
PYBIND11_PLUGIN(icu) {
    pybind11::module m("icu", "auto-compiled c++ extension");
    m.def("Modelsim", &Modelsim, py::return_value_policy::take_ownership);
    return m.ptr();
}
