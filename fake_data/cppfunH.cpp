#include <Rcpp.h>
#include <numeric>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <stdio.h>      /* printf */
#include <math.h>       /* lgamma */
using namespace Rcpp;
//[[Rcpp::plugins(cpp11)]]



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
 - PH   : the probability of Hospital admissions given infection;
 - etaH   : the overdispersion parameter.
 - pIC   : the probability of IC admissions given hospitalization;
 - etaIC   : the overdispersion parameter.
 
 gives as output:
 - yALL  : an integer vector (usually of size 66) 
 for the weekly number of Hospital [1:33]  and IC [34-66] admissions;
 
 ------------------------------------------------------------- */

// [[Rcpp::export]]
IntegerVector HandICUsim(NumericVector THETA){
  // set fixed elements:
  double spd     = 4;          // steps per day for the system of difference equations
  double begin   = 0;          // day of the begin of the epidemic
  double end     = 231;        // day of the end of the epidemic
  double dt      = 1/spd;      // time increment
  double N       = 55268100;   // population size
  double sgm     = 1;          // rate of moving E1->E2; E2->I1
  double gmm     = 0.52;       // rate of moving I1->I2; I2->r
  NumericVector zetaHt= { 0.23, 0.23, 0.24, 0.23, 0.19, 0.21,
                           0.20, 0.201, 0.202, 0.202, 0.203, 0.20,
                           0.193, 0.195, 0.199, 0.2, 0.19, 0.89,
                           0.21, 0.20, 0.21, 0.20, 0.20, 0.20,
                           0.22, 0.20, 0.201, 0.20, 0.203, 0.20,
                           0.21, 0.20, 0.20} ;  // vector of the detections of H
  NumericVector zetaICt= { 0.89, 0.89, 0.89, 0.89, 0.89, 0.89,
                         0.83, 0.75, 0.83, 0.89, 0.89, 0.89,
                         0.89, 0.89, 0.89, 0.89, 0.89, 0.89,
                         0.89, 0.89, 0.89, 0.89, 0.89, 0.89,
                         0.89, 0.89, 0.88, 0.8, 0.74, 0.89,
                         0.89, 0.86, 0.74} ;  // vector of the detections of IC
  NumericVector fEtoH={ 0.893541496,0.095125091,0.010126875,
                        0.001078092,0.000114772,0.000012218,
                        0.000001301,0.000000138,0.000000015,
                        0.000000002};       // probability of v weeks elapsing between 
                                            // infection and hosptal admissions 
  NumericVector fHtoIC= {0.939189937,0.057112199,0.003472996,
                         0.000211193,0.000012843,0.000000781,
                         0.000000047,0.000000003,0.000000000,
                         0.000000000};        // probability of v weeks elapsing between 
                                              // hospitalization and IC admissions   
  int sizemat=((end-begin)*spd)+1;  // epidemic model matrix size
  int lobs   = (sizemat/(7*spd));   // length of the final observations (weeks of the epidemics)
 
  // output vectors
  IntegerVector yALL(2*lobs);
  
  // parameters
  double beta  = THETA[0];
  double pi    = THETA[1];
  double iota  = THETA[2];
  double kappa = THETA[3];
  double pH    = THETA[4];
  double etaH  = THETA[5];
  double pIC   = THETA[6];
  double etaIC = THETA[7];
  // parameter existence check
  if( (pi+iota>=1)||(beta<=0)||(kappa<=-1)||(pH>=1)||(pH<=0)||(etaH<=1)||(pIC>=1)||(pIC<=0)||(etaIC<=1)){
    std::cout<< "One or more of the paramter values are impossible ";
    yALL = NA_INTEGER;
  }else{
    // set initial values of the system of difference equations:
    NumericMatrix mat(sizemat, 7);
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
    NumericVector NNI(lobs);
    for (int s=0;s<(lobs);s++){
      NNI[s] = mat(s*thin,1) - mat((s+1)*thin,1);
    }
    
    // number of new detected Hospital admissions obtained via convolution
    // and Negbinom
    NumericVector NH(lobs);
    NumericVector NNH(lobs);
    for (int s=0;s<(lobs);s++){
      double cumH=0;
      int R=std::min(s, 9);
      for (int r=0; r<(R+1); r++){
        cumH+=NNI[s-r]*fEtoH[r];
      }
      NH[s] = cumH*zetaHt[s]*pH;
      NNH[s] = cumH*pH;
      // obsevrations
      double sizeHs = NH[s]/(etaH-1); 
      yALL[s] = R::rnbinom(sizeHs, 1/(etaH) );
    }
    // number of new detected IC admissions obtained via convolution
    // and Negbinom
    NumericVector NIC(lobs); 
    for (int s=0;s<(lobs);s++){
      double cumIC=0;
      int R=std::min(s, 9);
      for (int r=0; r<(R+1); r++){
        cumIC+=NNH[s-r]*fHtoIC[r];
      }
      NIC[s] = cumIC*zetaICt[s]*pIC;
      // obsevrations
      double sizeICs = NIC[s]/(etaIC-1); 
      yALL[lobs+s] = R::rnbinom(sizeICs, 1/(etaIC) );
    }
  }
  return(yALL);
}

/*-----------------------------------------------------------
Function llICU:
computes the likelihood of a data vector, given parameter THETA

takes as a input: 
a. THETA a vector of size 6 with elements;
  - beta   : transmission rate
  - pi    : the proportion of initially immune;
  - iota  : the proportion of initially infected/infectious;
  - kappa : the factor for school holidays;
  - PH   : the probability of Hospital admissions given infection;
  - etaH   : the overdispersion parameter.
  - pIC   : the probability of IC admissions given hospitalization;
  - etaIC   : the overdispersion parameter.
 b. YH a vector of data
c. YIC a vector of data

 
gives as output:
- ll   : the log likelhood: log(p(YIC|THETA))

------------------------------------------------------------- */

// [[Rcpp::export]]
double llICU(NumericVector THETA, IntegerVector YH, IntegerVector YIC){
  // output ll element:
  double ll=0;
  
  // set fixed elements:
  int lobs=YIC.size();
  double spd     = 4;          // steps per day for the system of difference equations
  double begin   = 0;          // day of the begin of the epidemic
  double end     = lobs*7*spd; // day of the end of the epidemic data
  double dt      = 1/spd;      // time increment
  double N       = 55268100;   // population size
  double sgm     = 1;          // rate of moving E1->E2; E2->I1
  double gmm     = 0.52;       // rate of moving I1->I2; I2->r
  NumericVector zetaHt= { 0.23, 0.23, 0.24, 0.23, 0.19, 0.21,
                          0.20, 0.201, 0.202, 0.202, 0.203, 0.20,
                          0.193, 0.195, 0.199, 0.2, 0.19, 0.89,
                          0.21, 0.20, 0.21, 0.20, 0.20, 0.20,
                          0.22, 0.20, 0.201, 0.20, 0.203, 0.20,
                          0.21, 0.20, 0.20} ;  // vector of the detections of H
  NumericVector zetaICt= { 0.89, 0.89, 0.89, 0.89, 0.89, 0.89,
                           0.83, 0.75, 0.83, 0.89, 0.89, 0.89,
                           0.89, 0.89, 0.89, 0.89, 0.89, 0.89,
                           0.89, 0.89, 0.89, 0.89, 0.89, 0.89,
                           0.89, 0.89, 0.88, 0.8, 0.74, 0.89,
                           0.89, 0.86, 0.74} ;  // vector of the detections of IC
  NumericVector fEtoH={ 0.893541496,0.095125091,0.010126875,
                        0.001078092,0.000114772,0.000012218,
                        0.000001301,0.000000138,0.000000015,
                        0.000000002};       // probability of v weeks elapsing between 
                                            // infection and hosptal admissions 
  NumericVector fHtoIC= {0.939189937,0.057112199,0.003472996,
                         0.000211193,0.000012843,0.000000781,
                         0.000000047,0.000000003,0.000000000,
                         0.000000000};        // probability of v weeks elapsing between 
                                              // hospitalization and IC admissions   
  
  int sizemat=((end-begin)*spd)+1;  // epidemic model matrix size
  
  // parameters
  double beta  = THETA[0];
  double pi    = THETA[1];
  double iota  = THETA[2];
  double kappa = THETA[3];
  double pH    = THETA[4];
  double etaH  = THETA[5];
  double pIC   = THETA[6];
  double etaIC = THETA[7];
  // parameter existence check
  if( (pi+iota>=1)||(beta<=0)||(kappa<=-1)||(pH>=1)||(pH<=0)||(etaH<=1)||(pIC>=1)||(pIC<=0)||(etaIC<=1)){
    std::cout<< "One or more of the paramter values are impossible ";
    ll = - std::numeric_limits<float>::infinity();
  }else{
    // set initial values of the system of difference equations:
    NumericMatrix mat(sizemat, 7);
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
    NumericVector NNI(lobs);
    for (int s=0;s<(lobs);s++){
      NNI[s] = mat(s*thin,1) - mat((s+1)*thin,1);
    }
    
    // number of new detected Hoapital admissions obtained via convolution
    NumericVector NH(lobs);
    NumericVector NNH(lobs);
    for (int s=0;s<(lobs);s++){
      double cumH=0;
      int R=std::min(s, 9);
      for (int r=0; r<(R+1); r++){
        cumH+=NNI[s-r]*fEtoH[r];
      }
      NH[s] = cumH*zetaHt[s]*pH;
      NNH[s] = cumH*pH;
      
      // log likelihoods with cases
      double sizeHs = NH[s]/(etaH-1); 
      // if the mean is equal to zero and there are detected cases this is impossible 
      if((sizeHs==0) && (YH[s]!=0)){
        ll += -std::numeric_limits<float>::infinity();
      }else if((sizeHs==0) && (YH[s]==0)){
      // if the mean is equal to 0 and there are also zero cases this has probability 1
        ll +=0;
      }else{
      // negative binomial log likelihood
        ll += R::dnbinom(YH[s], sizeHs, 1/(etaH), true);
      }
    }
    
    // number of new detected ICU admissions obtained via convolution
    NumericVector NIC(lobs); 
    for (int s=0;s<(lobs);s++){
      double cumIC=0;
      int R=std::min(s, 9);
      for (int r=0; r<(R+1); r++){
        cumIC+=NNH[s-r]*fHtoIC[r];
      }
      NIC[s] = cumIC*zetaICt[s]*pIC;
      
      // log likelihoods with cases
      double sizeICs = NIC[s]/(etaIC-1); 
      // if the mean is equal to zero and there are detected cases this is impossible 
      if((sizeICs==0) && (YIC[s]!=0)){
        ll += -std::numeric_limits<float>::infinity();
      }else if((sizeICs==0) && (YIC[s]==0)){
        // if the mean is equal to 0 and there are also zero cases this has probability 1
        ll +=0;
      }else{
        // negative binomial log likelihood
        ll += R::dnbinom(YIC[s], sizeICs, 1/(etaIC), true);
      }
    }
  }
  return(ll);
}