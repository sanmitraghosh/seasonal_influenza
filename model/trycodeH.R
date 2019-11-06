# DIRECTORY AND LIBRARIES -------------------------------------------------

rm(list = ls())
# set this to the directory where you have your cppfun.cpp file
setwd("/Users/alicecorbella/Documents/sanmitra")  
library(Rcpp)
sourceCpp("cppfunH.cpp")

# SIMULATORS OF DATA ------------------------------------------------------

# paramter vetor
theta=c(0.53,      # beta
        0.30,      # pi
        0.00001, # iota  
        0.4,      # kappa
        0.0008,   # pH
        25,        # etaH                  
        0.07,      # pICgH
        15)        # etaIC

simALL <- HandICUsim(THETA=theta )
simyH  <- simALL[1:33]
simyIC <- simALL[34:66]

plot(simyH, type="l", ylim=c(0, 1000), col=3)
lines(simyIC, type="l", col=2)



for (s in 1:50){
  simALL <- HandICUsim(THETA=theta )
  simyH  <- simALL[1:33]
  simyIC <- simALL[34:66]
  
  lines(simyH, type="l", ylim=c(0, 1000), col=3)
  lines(simyIC, type="l", col=2)
}

rm(s, simyIC)

# choose 1 simulated dataset
set.seed(10519)
data <- HandICUsim(THETA= c(0.53, 0.30, 0.00001, 0.4, 
                        0.0008, 25,  0.07, 15) )
datayH  <- data[1:33]
datayIC <- data[34:66]
rm(data)
print(datayH)
print(datayIC)

# likelyhood function 
llICU(THETA=theta,YH=datayH, YIC = datayIC)


# try with other values
llICU(THETA=c(0.63, 0.30, 0.00001, 0.4,0.0008, 25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.40, 0.00001, 0.4,0.0008, 25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.000001, 0.4,0.0008, 25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.00001, 0.1,0.0008, 25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.00001, 0.4,0.008, 25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.00001, 0.4,0.0008, 15,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.00001, 0.4,0.0008, 25,  0.7, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.00001, 0.4,0.0008, 25,  0.07, 35), YH=datayH,YIC = datayIC)


# check with impossible value
llICU(THETA=c(-0.53, 0.30, 0.00001, 0.4,0.0008, 25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 1.30, 0.00001, 0.4,0.0008, 25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.71, 0.4,0.0008, 25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.00001, -1.4,0.0008, 25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.00001, 0.4,1.8, 25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.00001, 0.4,0.0008, .25,  0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.00001, 0.4,0.0008, 25,  -0.07, 15), YH=datayH,YIC = datayIC)
llICU(THETA=c(0.53, 0.30, 0.00001, 0.4,0.0008, 25,  0.07, -15), YH=datayH,YIC = datayIC)


