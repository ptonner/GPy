# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pylab as pb
from ..util.linalg import mdot, jitchol, chol_inv, pdinv
from ..util.plot import gpplot
from .. import kern
from GP import GP

#Still TODO:
# make use of slices properly (kernel can now do this)
# enable heteroscedatic noise (kernel will need to compute psi2 as a (NxMxM) array)

class sparse_GP(GP):
    """
    Variational sparse GP model

    :param X: inputs
    :type X: np.ndarray (N x Q)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP)
    :param kernel : the kernel/covariance function. See link kernels
    :type kernel: a GPy kernel
    :param X_uncertainty: The uncertainty in the measurements of X (Gaussian variance)
    :type X_uncertainty: np.ndarray (N x Q) | None
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (M x Q) | None
    :param Zslices: slices for the inducing inputs (see slicing TODO: link)
    :param M : Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type M: int
    :param normalize_(X|Y) : whether to normalize the data before computing (predictions will be in original scales)
    :type normalize_(X|Y): bool
    """

    def __init__(self, X, likelihood, kernel, Z, X_uncertainty=None, Xslices=None,Zslices=None, normalize_X=False):
        self.scale_factor = 1.0# a scaling factor to help keep the algorithm stable

        self.Z = Z
        self.Zslices = Zslices
        self.Xslices = Xslices
        self.M = Z.shape[0]
        self.likelihood = likelihood

        if X_uncertainty is None:
            self.has_uncertain_inputs=False
        else:
            assert X_uncertainty.shape==X.shape
            self.has_uncertain_inputs=True
            self.X_uncertainty = X_uncertainty

        GP.__init__(self, X, likelihood, kernel=kernel, normalize_X=normalize_X, Xslices=Xslices)

        #normalise X uncertainty also
        if self.has_uncertain_inputs:
            self.X_uncertainty /= np.square(self._Xstd)

    def _computations(self):
        # TODO find routine to multiply triangular matrices
        #TODO: slices for psi statistics (easy enough)

        sf = self.scale_factor
        sf2 = sf**2

        # kernel computations, using BGPLVM notation
        self.Kmm = self.kern.K(self.Z)
        if self.has_uncertain_inputs:
            self.psi0 = self.kern.psi0(self.Z,self.X, self.X_uncertainty)
            self.psi1 = self.kern.psi1(self.Z,self.X, self.X_uncertainty).T
            self.psi2 = self.kern.psi2(self.Z,self.X, self.X_uncertainty)
            if self.likelihood.is_heteroscedastic:
                self.psi2_beta_scaled = (self.psi2*(self.likelihood.precision.reshape(self.N,1,1)/sf2)).sum(0)
                #TODO: what is the likelihood is heterscedatic and there are multiple independent outputs?
            else:
                self.psi2_beta_scaled = (self.psi2*(self.likelihood.precision/sf2)).sum(0)
        else:
            self.psi0 = self.kern.Kdiag(self.X,slices=self.Xslices)
            self.psi1 = self.kern.K(self.Z,self.X)
            if self.likelihood.is_heteroscedastic:
                tmp = self.psi1*(np.sqrt(self.likelihood.precision.reshape(self.N,1))/sf)
            else:
                tmp = self.psi1*(np.sqrt(self.likelihood.precision)/sf)
            self.psi2_beta_scaled = np.dot(tmp,tmp.T)
            self.psi2 = self.psi1.T[:,:,None]*self.psi1.T[:,None,:] # TODO: remove me for efficiency and stability

        self.Kmmi, self.Lm, self.Lmi, self.Kmm_logdet = pdinv(self.Kmm)

        self.V = (self.likelihood.precision/self.scale_factor)*self.likelihood.Y
        self.A = mdot(self.Lmi, self.psi2_beta_scaled, self.Lmi.T)
        self.B = np.eye(self.M)/sf2 + self.A

        self.Bi, self.LB, self.LBi, self.B_logdet = pdinv(self.B)

        self.psi1V = np.dot(self.psi1, self.V)
        self.psi1VVpsi1 = np.dot(self.psi1V, self.psi1V.T)
        self.C = mdot(self.Lmi.T, self.Bi, self.Lmi)
        self.E = mdot(self.C, self.psi1VVpsi1/sf2, self.C.T)

        # Compute dL_dpsi # FIXME: this is untested for the het. case
        self.dL_dpsi0 = - 0.5 * self.D * self.likelihood.precision * np.ones(self.N)
        self.dL_dpsi1 = mdot(self.V, self.psi1V.T,self.C).T
        if self.likelihood.is_heteroscedastic:
            self.dL_dpsi2 = 0.5 * self.likelihood.precision[:,None,None] * self.D * self.Kmmi[None,:,:] # dB
            self.dL_dpsi2 += - 0.5 * self.likelihood.precision[:,None,None]/sf2 * self.D * self.C[None,:,:] # dC
            self.dL_dpsi2 += - 0.5 * self.likelihood.precision[:,None,None]* self.E[None,:,:] # dD
        else:
            self.dL_dpsi2 = 0.5 * self.likelihood.precision * self.D * self.Kmmi[None,:,:] # dB
            self.dL_dpsi2 += - 0.5 * self.likelihood.precision/sf2 * self.D * self.C[None,:,:] # dC
            self.dL_dpsi2 += - 0.5 * self.likelihood.precision * self.E[None,:,:] # dD

        # Compute dL_dKmm
        self.dL_dKmm = -0.5 * self.D * mdot(self.Lmi.T, self.A, self.Lmi)*sf2 # dB
        self.dL_dKmm += -0.5 * self.D * (- self.C/sf2 - 2.*mdot(self.C, self.psi2_beta_scaled, self.Kmmi) + self.Kmmi) # dC
        self.dL_dKmm +=  np.dot(np.dot(self.E*sf2, self.psi2_beta_scaled) - np.dot(self.C, self.psi1VVpsi1), self.Kmmi) + 0.5*self.E # dD

        #the partial derivative vector for the likelihood
        self.partial_for_likelihood = - 0.5 * self.D*self.likelihood.precision + 0.5 * (self.likelihood.Y**2).sum(1)*self.likelihood.precision**2 #dA
        self.partial_for_likelihood +=  0.5 * self.D * (self.psi0*self.likelihood.precision**2 - (self.psi2*self.Kmmi[None,:,:]*self.likelihood.precision[:,None,None]**2).sum(1).sum(1)/sf2) #dB
        #self.partial_for_likelihood +=  0.5 * self.D * np.sum(self.Bi*self.A)*self.likelihood.precision #dC
        #self.partial_for_likelihood += -np.diag(np.dot((self.C - 0.5 * mdot(self.C,self.psi2_beta_scaled,self.C) ) , self.psi1VVpsi1 ))*self.likelihood.precision #dD


    def _set_params(self, p):
        self.Z = p[:self.M*self.Q].reshape(self.M, self.Q)
        self.kern._set_params(p[self.Z.size:self.Z.size+self.kern.Nparam])
        self.likelihood._set_params(p[self.Z.size+self.kern.Nparam:])
        self._computations()

    def _get_params(self):
        return np.hstack([self.Z.flatten(),GP._get_params(self)])

    def _get_param_names(self):
        return sum([['iip_%i_%i'%(i,j) for i in range(self.Z.shape[0])] for j in range(self.Z.shape[1])],[]) + GP._get_param_names(self)

    def log_likelihood(self):
        """ Compute the (lower bound on the) log marginal likelihood """
        sf2 = self.scale_factor**2
        if self.likelihood.is_heteroscedastic:
            A = -0.5*self.N*self.D*np.log(2.*np.pi) +0.5*np.sum(np.log(self.likelihood.precision)) -0.5*np.sum(self.V*self.likelihood.Y)
        else:
            A = -0.5*self.N*self.D*(np.log(2.*np.pi) - np.log(self.likelihood.precision)) -0.5*self.likelihood.precision*self.likelihood.trYYT
        B = -0.5*self.D*(np.sum(self.likelihood.precision*self.psi0) - np.trace(self.A)*sf2)
        C = -0.5*self.D * (self.B_logdet + self.M*np.log(sf2))
        D = +0.5*np.sum(self.psi1VVpsi1 * self.C)
        return A+B+C+D

    def _log_likelihood_gradients(self):
        return np.hstack((self.dL_dZ().flatten(), self.dL_dtheta(), self.likelihood._gradients(partial=self.partial_for_likelihood)))

    def dL_dtheta(self):
        """
        Compute and return the derivative of the log marginal likelihood wrt the parameters of the kernel
        """
        dL_dtheta = self.kern.dK_dtheta(self.dL_dKmm,self.Z)
        if self.has_uncertain_inputs:
            dL_dtheta += self.kern.dpsi0_dtheta(self.dL_dpsi0, self.Z,self.X,self.X_uncertainty)
            dL_dtheta += self.kern.dpsi1_dtheta(self.dL_dpsi1.T,self.Z,self.X, self.X_uncertainty)
            dL_dtheta += self.kern.dpsi2_dtheta(self.dL_dpsi2,self.dL_dpsi1.T, self.Z,self.X, self.X_uncertainty)
        else:
            #re-cast computations in psi2 back to psi1:
            dL_dpsi1 = self.dL_dpsi1 + 2.*np.dot(self.dL_dpsi2.sum(0),self.psi1)
            dL_dtheta += self.kern.dK_dtheta(dL_dpsi1,self.Z,self.X)
            dL_dtheta += self.kern.dKdiag_dtheta(self.dL_dpsi0, self.X)

        return dL_dtheta

    def dL_dZ(self):
        """
        The derivative of the bound wrt the inducing inputs Z
        """
        dL_dZ = 2.*self.kern.dK_dX(self.dL_dKmm,self.Z)#factor of two becase of vertical and horizontal 'stripes' in dKmm_dZ
        if self.has_uncertain_inputs:
            dL_dZ += self.kern.dpsi1_dZ(self.dL_dpsi1,self.Z,self.X, self.X_uncertainty)
            dL_dZ += 2.*self.kern.dpsi2_dZ(self.dL_dpsi2,self.Z,self.X, self.X_uncertainty) # 'stripes'
        else:
            #re-cast computations in psi2 back to psi1:
            dL_dpsi1 = self.dL_dpsi1 + 2.*np.dot(self.dL_dpsi2.sum(0),self.psi1)
            dL_dZ += self.kern.dK_dX(dL_dpsi1,self.Z,self.X)
        return dL_dZ

    def _raw_predict(self, Xnew, slices, full_cov=False):
        """Internal helper function for making predictions, does not account for normalisation"""

        Kx = self.kern.K(self.Z, Xnew)
        mu = mdot(Kx.T, self.C/self.scale_factor, self.psi1V)

        if full_cov:
            Kxx = self.kern.K(Xnew)
            var = Kxx - mdot(Kx.T, (self.Kmmi - self.C/self.scale_factor**2), Kx)
        else:
            Kxx = self.kern.Kdiag(Xnew)
            var = Kxx - np.sum(Kx*np.dot(self.Kmmi - self.C/self.scale_factor**2, Kx),0)

        return mu,var

    def plot(self, *args, **kwargs):
        """
        Plot the fitted model: just call the GP_regression plot function and then add inducing inputs
        """
        GP_regression.plot(self,*args,**kwargs)
        if self.Q==1:
            pb.plot(self.Z,self.Z*0+pb.ylim()[0],'k|',mew=1.5,markersize=12)
            if self.has_uncertain_inputs:
                pb.errorbar(self.X[:,0], pb.ylim()[0]+np.zeros(self.N), xerr=2*np.sqrt(self.X_uncertainty.flatten()))
        if self.Q==2:
            pb.plot(self.Z[:,0],self.Z[:,1],'wo')
