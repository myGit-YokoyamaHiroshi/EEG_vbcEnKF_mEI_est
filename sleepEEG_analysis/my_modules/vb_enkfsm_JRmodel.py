#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:49:09 2021

@author: user
"""

import numpy as np
from copy import deepcopy
from scipy.linalg import sqrtm, cholesky, lu
from scipy.special import gamma, digamma, gammaln
from numpy.linalg import slogdet

class vbEnKF_JansenRit:
    def __init__(self, X, P, Q, R, dt, Npar=200):
        ###############
        self.X     = X
        self.P     = P
        self.Q     = Q
        self.R     = R
        self.dt    = dt
        self.Npar  = Npar
        # self.a0    = 1
        # self.b0    = 0.5
        self.a0    = 1E-3
        self.b0    = 1E-3
        
        self.a     = self.a0# + 1/2
        self.b     = self.b0
    
    def Sigm(self, v):
        v0   = 6
        vmax = 5
        r    = 0.56
        sigm = vmax / (1 + np.exp(r * ( v0 - v )))
        
        return sigm
    
    def postsynaptic_potential_function(self, y, z, A, a, Sgm):
        dy = z
        dz = A * a * Sgm - 2 * a * z - a**2 * y
        
        f_out = np.hstack((dy, dz))
        return f_out
    
    def JansenRit_model(self, x, par):
        dt   = self.dt
        X    = np.hstack((x, par))
        
        A    = par[0]
        a    = par[1]
        B    = par[2]
        b    = par[3]
        u    = par[4]
        
        dx   = np.zeros(len(x))
        C    = 135
        c1   = 1.0  * C
        c2   = 0.8  * C
        c3   = 0.25 * C
        c4   = 0.25 * C
        
        Sgm_12 = self.Sigm(x[1] - x[2]);
        Sgm_p0 = u + c2 * self.Sigm(c1*x[0]);
        Sgm_0  = c4 * self.Sigm(c3*x[0]);
            
        dx_03 = self.postsynaptic_potential_function(x[0], x[3], A, a, Sgm_12);
        dx_14 = self.postsynaptic_potential_function(x[1], x[4], A, a, Sgm_p0);
        dx_25 = self.postsynaptic_potential_function(x[2], x[5], B, b, Sgm_0);
        
        # sort order of dy
        dx[0] = dx_03[0]
        dx[3] = dx_03[1]
        
        dx[1] = dx_14[0]
        dx[4] = dx_14[1]
        
        dx[2] = dx_25[0]
        dx[5] = dx_25[1]
        
        dX    = np.hstack((dx, np.zeros(par.shape)))
        return dX
    
    def state_func(self, x, par):
        dt    = self.dt
        X_now = np.hstack((x, par))
        
        k1   = self.JansenRit_model(X_now[:6], X_now[6:])

        X_k2 = X_now + (dt/2)*k1
        k2   = self.JansenRit_model(X_k2[:6], X_k2[6:])
        
        X_k3 = X_now + (dt/2)*k2
        k3   = self.JansenRit_model(X_k3[:6], X_k3[6:])
        
        X_k4 = X_now + dt*k3
        k4   = self.JansenRit_model(X_k4[:6], X_k4[6:])
        
        X_next = X_now + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        return X_next
    
    
    def inv_lu(self, X):
        p,l,u = lu(X, permute_l = False)
        l     = np.dot(p,l) 
        l_inv = np.linalg.inv(l)
        u_inv = np.linalg.inv(u)
        X_inv = np.dot(u_inv,l_inv)
        
        return X_inv

    ###################
    def predict(self):
        X       = self.X
        P       = self.P
        Q       = self.Q
        dt      = self.dt
        Npar    = self.Npar
        Nstate  = len(X)
        
        
        x_sgm   = np.random.multivariate_normal(mean=X, cov=P, size=Npar)
        v       = np.random.multivariate_normal(mean=np.zeros(len(X)), cov=Q, size=Npar)
        X_sgm   = np.array([self.state_func(x_sgm[i,:6], x_sgm[i,6:]) + v[i] for i in range(Npar)])
        # X_sgm   = np.array([self.state_func(x_sgm[i,:6], x_sgm[i,6:])  for i in range(Npar)])
        XPred   = np.mean(X_sgm, axis=0)  
        
        self.X_sgm_ = x_sgm
        self.X_     = X
        self.P_     = P
        
        dx          = X_sgm.T - XPred[:, np.newaxis]
        PPred       = ((dx @ dx.T)/(Npar-1)) + Q
        
        self.X      = XPred
        self.P      = PPred
        self.X_sgm  = X_sgm
    
    def smoother(self, Xsgm_lag, X_now, P_now, Xsgm_now, H, z, z_sgm, w):
        Xsgm_lag = self.Xsgm_lag        
        R        = self.R
        Npar     = self.Npar
        Nstate   = len(X_now)
        a        = self.a
        b        = self.b
        weight   = b/a
        
        Nlag     = Xsgm_lag.shape[2]
        dx       = Xsgm_now.T - X_now[:, np.newaxis]        
        dX_      = Xsgm_lag - np.mean(Xsgm_lag, axis=0)[np.newaxis, :]
        Pxx_     = np.array([(dX_[:,:,lg].T @ dx.T)/(Npar-1) for lg in range(Nlag)])
        Pxx_     = Pxx_.reshape((Nlag*Nstate, Nstate))
            
        Delta    = Pxx_ @ H.T @ np.linalg.inv(H @ P_now @ H.T + (weight * R))
        
        Xtmp     = Xsgm_lag.reshape((Npar, Nstate*Nlag), order='f').T
        Xsm      = Xtmp + Delta @ (z + w - z_sgm)
        
        Xsgm_lag = (Xsm.T).reshape(Xsgm_lag.shape, order='f')
            
        return Xsgm_lag
    
    def update(self):
        z     = self.z
        X     = self.X
        X_sgm = self.X_sgm
        P     = self.P
        R     = self.R
        Npar  = self.Npar
        dt    = self.dt
        a     = self.a + 1/2
        b     = self.b
        
        eta   = b/a
        
        Xsgm_lag = self.Xsgm_lag
        
        D     = np.array([
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                         ])
        ub    = np.array([100.00,  200, 100.00,  200, 320])
        lb    = np.array([  0.01,    5,   0.01,    5, 120])
        c     = np.zeros(ub.shape)
        
        H     = np.array([[0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]])
        ################
        z_sgm = H @ X_sgm.T 
        zPred = np.mean(z_sgm ,axis=1)  
        y     = z - zPred # prediction error of observation model
        
        #################################################################
        dx      = X_sgm.T - X[:, np.newaxis]
        dz      = z_sgm - zPred
        
        Pxz     = (dx @ dz.T)/(Npar-1)
        Pzz     = ((dz @ dz.T )/(Npar-1)) + (eta*R)
        Pzz_inv = np.linalg.inv(Pzz)
        #%%        
        w       = np.random.normal(loc=0, scale=eta*R, size=Npar)
        K       = Pxz @ Pzz_inv # Kalman Gain
        X_new   = np.mean(X_sgm.T + K@(z + w - z_sgm),axis=1)
        P_new   = P - K @ Pzz @ K.T
        
        if np.isnan(Xsgm_lag).all()==False:
            Xsgm_lag = self.smoother(Xsgm_lag, X, P, X_sgm, H, z, z_sgm, w)
            self.Xsgm_lag = Xsgm_lag
        
        b       = b + 1/2 * ((z - (H@X_new))**2)/R + 1/2 * np.trace((H@P_new@H.T)/R)
        ##### inequality constraints ##########################################
        ###   Constraint would be applied only when the inequality condition is not satisfied.
        I     = np.eye(len(X_new))
        W_inv = np.linalg.inv(P_new)
        L     = W_inv @ D.T @ np.linalg.inv(D @ W_inv @ D.T)
        value = D @ X_new 
        for i in range(len(value)):
            if (value[i] > ub[i]) | (value[i] < lb[i]):
                if (value[i] > ub[i]): 
                    c[i] = ub[i]
                elif (value[i] < lb[i]):
                    c[i] = lb[i]
        ## Calculate state variables with interval contraints
        X_c   = X_new - L @ (D @ X_new - c)
        for i in range(len(value)):
            if (value[i] > ub[i]) | (value[i] < lb[i]):
                X_new[i+6] = X_c[i+6]
        ##### inequality constraints ##########################################
        
        ####### loglike
        logdetR    = np.log(R)
        s, logdetP = slogdet(P)
        P_inv      = np.linalg.inv(P)
        err        = z - H @ X
        mu         = X_new - X#
        Nstate     = len(X)
        N          = len(zPred)
        
        loglike    = -0.5 * (     N * np.log(2*np.pi) + np.log(eta*R) + a/b * np.sum(err**2)/R) \
                     -0.5 * (Nstate * np.log(2*np.pi) + logdetP + mu.T @ P_inv @ mu)
        ####### elbo
        ll_state   = 1/2 * (-Nstate * np.log(2*np.pi) - logdetP - np.trace(P_inv@P))
        ll_obs     = 1/2 * (-N * np.log(2*np.pi) +  (digamma(a) - np.log(b)) - logdetR - a/b * ((np.sum(err**2/R)) + np.trace((H@P@H.T)/R)))
        ll_gamma   = (self.a-1)* (digamma(a)-np.log(b))-gammaln(self.a)+self.a*np.log(self.b)-self.b*(a/b)

        Hx         = Nstate/2 * (1 + np.log(2*np.pi)) + 1/2 * logdetP
        Heta       = a - np.log(b) + gammaln(a) + (1-a) * digamma(a)
        
        ELBO       = ll_state + ll_obs + ll_gamma + Hx + Heta
        
        
        self.X       = X_new
        self.P       = P_new
        self.zPred   = zPred
        self.S       = Pzz
        self.elbo = ELBO
        self.a       = a
        self.b       = b
    
    def vbenkf_estimation(self, z, Xsgm_lag=np.nan):   
        self.z = z
        self.Xsgm_lag = Xsgm_lag
        # Prediction step (estimate state variable)
        self.predict()
        
        # Update state (Update parameters)
        self.update()
