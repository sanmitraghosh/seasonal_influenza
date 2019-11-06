# DIRECTORY AND LIBRARIES -------------------------------------------------

rm(list = ls())
# set this to the directory where you have your cppfun.cpp file
setwd("/scratch/sanmitra/AH1N1_2011")  
library(Rcpp)
sourceCpp("cppfun.cpp")

# SIMULATORS OF DATA ------------------------------------------------------

# paramter vetor
theta=c(0.56,      # beta
        0.36,      # pi
        0.0000061, # iota  
        0.37,      # kappa
        0.00015,   # pIC
        15)        # eta                  


simYic <- ICUsim(THETA=theta )

plot(simYic, type="l", ylim=c(0, 600))
for (s in 1:50){
  simYic <- ICUsim(THETA= c(0.56, 0.36, 0.0000061, 0.37, 0.00015, 15))
  lines(simYic)
}

rm(s, simYic)

# choose 1 simulated dataset
set.seed(10519)
# data <- ICUsim(THETA= c(0.56, 0.36, 0.0000051, 0.37, 0.00015,15))
data <- ICUsim(THETA= c(0.56, 0.36, 0.0000051, -0.35, 0.00015,15))
print(data)

# likelyhood function 
llICU(THETA=theta, YIC = data)


# try with other values
llICU(THETA=c(0.66, 0.36, 0.0000061, 0.37, 0.00015, 15), YIC = data)
llICU(THETA=c(0.56, 0.56, 0.0000061, 0.37, 0.00015, 15), YIC = data)
llICU(THETA=c(0.56, 0.36, 0.00001, 0.37, 0.00015, 15), YIC = data)
llICU(THETA=c(0.56, 0.36, 0.0000061, 0.5, 0.00015, 15), YIC = data)
llICU(THETA=c(0.56, 0.36, 0.0000061, 0.37, 0.0003, 15), YIC = data)
llICU(THETA=c(0.56, 0.36, 0.0000061, 0.37, 0.00015, 2), YIC = data)


# check with impossible value
llICU(THETA=c(-0.56, 0.36, 0.0000061, 0.37, 0.00015, 15), YIC = data)
llICU(THETA=c(0.56, 1.36, 0.0000061, 0.37, 0.00015, 15), YIC = data)
llICU(THETA=c(0.56, 0.36, 0.71, 0.37, 0.00015, 15), YIC = data)
llICU(THETA=c(0.56, 0.36, 0.0000061, -1.37, 0.00015, 15), YIC = data)
llICU(THETA=c(0.56, 0.36, 0.0000061, 0.37, 1.5, 15), YIC = data)
llICU(THETA=c(0.56, 0.36, 0.0000061, 0.37, 0.00015, 0.15), YIC = data)
