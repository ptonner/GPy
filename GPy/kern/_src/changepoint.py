from .kern import Kern, CombinationKernel
from ...core.parameterization import Param
from ...core.parameterization.transformations import Logexp
import numpy as np

class Changepoint(CombinationKernel):
    """Kernel for a changepoint at position xc """
    
    def __init__(self,k1,k2,kc,xc,cpDim):
        if k2 is None:
            super(Changepoint,self).__init__([k1],"changepoint")
            k2 = k1
        else:
            super(Changepoint,self).__init__([k1,k2],"changepoint")
        
        self.k1 = k1
        self.k2 = k2
        
        self.kc = Param('kc', kc, Logexp())
        self.link_parameter(self.kc)
        
        self.xc = xc
        self.cpDim = cpDim
        
    def Kdiag(self,X):
        xside = X[:,self.cpDim] < self.xc[:,self.cpDim]
        
        K1 = self.k1.Kdiag(X)
        K2 = self.k2.Kdiag(X)
        
        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)
        
        G1 = self.k1.K(X,self.xc) / n1
        G2 = self.k2.K(X,self.xc) / n2
        
        return np.where(xside,K1 + G1*G1*(self.kc-n1),K2 + G2*G2*(self.kc-n2))
    
    def K(self,X,X2=None):
        
        if X2 is None:
            X2 = X
        
        K1 = self.k1.K(X,X2)
        K2 = self.k2.K(X,X2)
        
        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)
        
        G11 = self.k1.K(X,self.xc) / n1
        G12 = self.k1.K(X2,self.xc) / n1
        G21 = self.k2.K(X,self.xc) / n2
        G22 = self.k2.K(X2,self.xc) / n2
        
        x1side = X[:,self.cpDim] < self.xc[:,self.cpDim]
        x1side_2 = X[:,self.cpDim] > self.xc[:,self.cpDim]
        x2side = X2[:,self.cpDim] < self.xc[:,self.cpDim]
        x2side_2 = X2[:,self.cpDim] > self.xc[:,self.cpDim]
        
        k = np.where( np.outer(x1side,x2side),K1 + np.dot(G11,G12.T)*(self.kc-n1),
                         np.where(np.outer(x1side_2,x2side_2), K2 + np.dot(G21,G22.T)*(self.kc-n2),
                                  np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T)*self.kc,
                                           np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T)*self.kc, 0
                         ))))
        
        return k
    
    def update_gradients_full(self, dL_dK, X, X2=None):
        """"""
        
        if X2 is None:
            X2 = X
        
        k = self.K(X,X2)*dL_dK
        
        x1side = X[:,self.cpDim] < self.xc[:,self.cpDim]
        x1side_2 = X[:,self.cpDim] > self.xc[:,self.cpDim]
        x2side = X2[:,self.cpDim] < self.xc[:,self.cpDim]
        x2side_2 = X2[:,self.cpDim] > self.xc[:,self.cpDim]
        
        n1 = self.k1.K(self.xc,self.xc)
        n2 = self.k2.K(self.xc,self.xc)
        
        G11 = self.k1.K(X,self.xc) / n1
        G12 = self.k1.K(X2,self.xc) / n1
        G21 = self.k2.K(X,self.xc) / n2
        G22 = self.k2.K(X2,self.xc) / n2
        
        # dL_dK1 = dL_dK if X,X2 < xc:
        self.k1.update_gradients_full(np.where(np.outer(x1side,x2side),dL_dK,0),X,X2)
        
        # dL_dK2 = dL_dK if X,X2 > xc:
        self.k2.update_gradients_full(np.where(np.outer(x1side_2,x2side_2),dL_dK,0),X,X2)
        
        
        self.kc.gradient = np.sum(dL_dK*
                np.where( np.outer(x1side,x2side),np.dot(G11,G12.T),
                         np.where(np.outer(x1side_2,x2side_2), np.dot(G21,G22.T),
                                  np.where(np.outer(x1side,x2side_2), np.dot(G11,G22.T),
                                           np.where(np.outer(x1side_2,x2side), np.dot(G21,G12.T), 0
                         )))))