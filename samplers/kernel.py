"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.
Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

from abc import abstractmethod
from numpy import eye, concatenate, zeros, shape, mean, reshape, arange, exp, outer, median, sqrt
from numpy.random import permutation,shuffle
from numpy.lib.index_tricks import fill_diagonal
from matplotlib.pyplot import imshow,show
from scipy.spatial.distance import squareform, pdist, cdist
import numpy as np
from numba import jit, generated_jit, types
import scipy

class Kernel(object):
    def __init__(self):
        pass

    @abstractmethod
    def kernel(self, X, Y=None):
        raise NotImplementedError()
    
    @abstractmethod
    def gradient(self, x, Y):
        
        # ensure this in every implementation
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        raise NotImplementedError()
    
    @staticmethod
    def centring_matrix(n):
        """
        Returns the centering matrix eye(n) - 1.0 / n
        """
        return eye(n) - 1.0 / n
    
    @staticmethod
    def center_kernel_matrix(K):
        """
        Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
        """
        n = shape(K)[0]
        H = eye(n) - 1.0 / n
        return  1.0 / n * H.dot(K.dot(H))
    
    @abstractmethod
    def show_kernel_matrix(self,X,Y=None):
        K=self.kernel(X,Y)
        imshow(K, interpolation="nearest")
        show()
    
    @abstractmethod
    def estimateMMD(self,sample1,sample2,unbiased=False):
        """
        Compute the MMD between two samples
        """
        K11 = self.kernel(sample1,sample1)
        K22 = self.kernel(sample2,sample2)
        K12 = self.kernel(sample1,sample2)
        if unbiased:
            fill_diagonal(K11,0.0)
            fill_diagonal(K22,0.0)
            n=float(shape(K11)[0])
            m=float(shape(K22)[0])
            return sum(sum(K11))/(pow(n,2)-n) + sum(sum(K22))/(pow(m,2)-m) - 2*mean(K12[:])
        else:
            return mean(K11[:])+mean(K22[:])-2*mean(K12[:])
        
class GaussianKernel(Kernel):
    def __init__(self, sigma):
        Kernel.__init__(self)
        
        self.width = sigma
    
    def kernel(self, X, Y=None):
        """
        Computes the standard Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)
        
        X - 2d array, samples on right hand side
        Y - 2d array, samples on left hand side, can be None in which case its replaced by X
        """
        
        #assert(len(np.shape(X))==2)
        
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            sq_dists = scipy.spatial.distance.squareform(pdist(X, 'sqeuclidean'))
        else:
            assert(len(np.shape(Y))==2)
            assert(np.shape(X)[1]==np.shape(Y)[1])
            sq_dists = scipy.spatial.distance.cdist(X, Y, 'sqeuclidean')
    
        K = np.exp(-0.5 * (sq_dists) / self.width ** 2)
        return K
    
    def gradient(self, x, Y):
        """
        Computes the gradient of the Gaussian kernel wrt. to the left argument, i.e.
        k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2), which is
        \nabla_x k(x,y)=1.0/sigma**2 k(x,y)(y-x)
        Given a set of row vectors Y, this computes the
        gradient for every pair (x,y) for y in Y.
        
        x - single sample on right hand side (1D vector)
        Y - samples on left hand side (2D matrix) N X D
        """
        assert(len(np.shape(x))==1)
        assert(len(np.shape(Y))==2)
        assert(len(x)==np.shape(Y)[1])
        
        x_2d=np.reshape(x, (1, len(x))) #  1 X D
        k = self.kernel(x_2d, Y) #  1 X N
        differences = Y - x #  N X D
        G = (1.0 / self.width ** 2) * (k.T * differences) #  N X D
        return G



    @staticmethod
    def get_sigma_median_heuristic(Z, num_subsample=5000):
        
        inds = np.random.permutation(len(Z))[:np.max([num_subsample, len(Z)])]
        dists = squareform(pdist(Z[inds], 'sqeuclidean'))
        median_dist = np.median(dists[dists > 0])
        sigma = np.sqrt(0.5 * median_dist)
        #gamma = 0.5 / (sigma ** 2)
        
        return sigma
        
    @staticmethod
    def get_sigma_median_heuristic_twosample(sample1, sample2, num_subsample=1000):

        assert(shape(sample1)[1]==shape(sample1)[1])
          
        inds = np.random.permutation(len(sample1))[:np.max([num_subsample, len(sample1)])]
        sq_dists = cdist(sample1[inds], sample2[inds], 'sqeuclidean')      
        median_dist = np.median(sq_dists[sq_dists > 0])
        sigma = np.sqrt(0.5 * median_dist)
        #gamma = 0.5 / (sigma ** 2)
        
        return sigma

class MaternKernel(Kernel):
    def __init__(self, rho, nu=1.5, sigma=1.0):
        Kernel.__init__(self)
        #GenericTests.check_type(rho,'rho',float)
        
        self.rho = rho
        self.nu = nu
        self.sigma = sigma
    def kernel(self, X, Y=None):
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            dists = squareform(pdist(X, 'euclidean'))
        else:
            assert(shape(X)[1]==shape(Y)[1])
            dists = cdist(X, Y, 'euclidean')
        if self.nu==0.5:
            #for nu=1/2, Matern class corresponds to Ornstein-Uhlenbeck Process
            K = (self.sigma**2.) * exp( -dists / self.rho )                 
        elif self.nu==1.5:
            K = (self.sigma**2.) * (1+ sqrt(3.)*dists / self.rho) * exp( -sqrt(3.)*dists / self.rho )
        elif self.nu==2.5:
            K = (self.sigma**2.) * (1+ sqrt(5.)*dists / self.rho + 5.0*(dists**2.) / (3.0*self.rho**2.) ) * exp( -sqrt(5.)*dists / self.rho )
        else:
            raise NotImplementedError()
        return K
    
    def gradient(self, x, Y):
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        if self.nu==1.5 or self.nu==2.5:
            x_2d=reshape(x, (1, len(x)))
            lower_order_rho = self.rho * sqrt(2*(self.nu-1)) / sqrt(2*self.nu)
            lower_order_kernel = MaternKernel(lower_order_rho,self.nu-1,self.sigma)
            k = lower_order_kernel.kernel(x_2d, Y)
            differences = Y - x
            G = ( 1.0 / lower_order_rho ** 2 ) * (k.T * differences)
            return G
        else:
            raise NotImplementedError()

class GaussianARDKernel(Kernel):
    def __init__(self, sigmas):
        Kernel.__init__(self)
        ells = np.array(sigmas)
        self.width = np.diag(1./ells**2)
    
    def kernel(self, X, Y=None):
        """
        Computes the standard Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)
        
        X - 2d array, samples on right hand side
        Y - 2d array, samples on left hand side, can be None in which case its replaced by X
        """
        
        assert(len(shape(X))==2)
        
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            sq_dists = squareform(pdist(X, 'sqeuclidean'))
        else:
            assert(len(shape(Y))==2)
            assert(shape(X)[1]==shape(Y)[1])
            Xll = X.dot(self.width)
            Yll = Y.dot(self.width)
            sq_dists = cdist(Xll, Yll, 'sqeuclidean')
    
        K = exp(-0.5 * (sq_dists) )
        return K
    
    def gradient(self, x, Y):
        """
        Computes the gradient of the Gaussian kernel wrt. to the left argument, i.e.
        k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2), which is
        \nabla_x k(x,y)=1.0/sigma**2 k(x,y)(y-x)
        Given a set of row vectors Y, this computes the
        gradient for every pair (x,y) for y in Y.
        
        x - single sample on right hand side (1D vector)
        Y - samples on left hand side (2D matrix)
        """
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        x_2d=reshape(x, (1, len(x)))
        Xll = x_2d.dot(self.width)
        Yll = Y.dot(self.width)        
        k = self.kernel(Xll, Yll)
        differences = Yll - Xll.reshape((len(x),))
        G = (1.0) * (k.T * differences)
        return G
class SumKernel(Kernel):
    def __init__(self, list_of_kernels):
        Kernel.__init__(self)
        self.list_of_kernels = list_of_kernels
        
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        return np.sum([individual_kernel.kernel(X,Y) for individual_kernel in self.list_of_kernels],0)
    def gradient(self, x, Y):
        return np.sum([individual_kernel.gradient(x,Y) for individual_kernel in self.list_of_kernels],0)