from .kern import Kern, CombinationKernel
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
import numpy as np


class Changepoint(CombinationKernel):
    """Kernel for points across a changepoint. K(X,Y) = kc * K1(X,cp) * K2(Y,cp) """
    
    def __init__(self,k1,k2,cpDim,kc=1,**kwargs):
        super(Changepoint,self).__init__([k1,k2],"changepoint")
        self.cpDim = cpDim
        self.cp = Param('cp',0)
        self.link_parameter(self.cp)
        self.kc = Param('kc',kc,Logexp())
        self.link_parameter(self.kc)
        
    def K(self,X,X2=None):
        if X2 is None:
            X2 = X

        pos = np.zeros((X.shape[0],X2.shape[0],2))
        pos[:,:,0] = np.repeat(X,X2.shape[0],1)
        pos[:,:,1] = np.repeat(X2,X.shape[0],1).T

        return np.where(pos[:,:,0] < self.cp,
                    np.where(pos[:,:,1] < self.cp, self.parts[0].K(X,X2), self.kc * np.outer(self.parts[0].K(np.array(self.cp)[:,None],X),self.parts[1].K(np.array(self.cp)[:,None],X2))),
                    np.where(pos[:,:,1] > self.cp, self.parts[1].K(X,X2), self.kc * np.outer(self.parts[1].K(np.array(self.cp)[:,None],X), self.parts[0].K(np.array(self.cp)[:,None],X2))))

        # return np.where(X[:,self.cpDim]<self.cp,
        #                     np.where(X2[:,self.cpDim]<self.cp, self.parts[0].K(X,X2), self.kc), 
        #                     np.where(X2[:,self.cpDim]>self.cp, self.parts[1].K(X,X2), self.kc))
        
        # return np.where(
        #     np.all((X[:,self.cpDim]<self.cp,X2[:,self.cpDim]<self.cp),0), self.parts[0].K(X,X2),
        #         np.where(np.all((X[:,self.cpDim]>self.cp,X2[:,self.cpDim]>self.cp),0), self.parts[1].K(X,X2),
        #             self.kc
        #             ))
    
    def Kdiag(self,X):            
        return np.diag(self.K(X))
    
    def update_gradients_full(self, dL_dK, X, X2=None):
        print "cp_cross update_gradients_full"
        return
        k = self.K(X,X2)*dL_dK
#         try:
        for p in self.parts:
            if isinstance(p,GPy.kern.Kern):
                p.update_gradients_full(k/p.K(X,X2),X,X2)
#         except FloatingPointError:
#             for combination in itertools.combinations(self.parts, len(self.parts) - 1):
#                 prod = reduce(np.multiply, [p.K(X, X2) for p in combination])
#                 to_update = list(set(self.parts) - set(combination))[0]
#                 to_update.update_gradients_full(dL_dK * prod, X, X2)
