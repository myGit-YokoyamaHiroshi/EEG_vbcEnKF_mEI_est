#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:49:09 2021

@author: user
"""

import numpy as np
from copy import deepcopy
from scipy.linalg import sqrtm, cholesky


class UKF_JansenRit:
    def __init__(self, X, P, Q, R, dt):
        nx      = len(X)
        ALPHA   = 1E-4
        BETA    = 2
        KAPPA   = 0
        
        lamda = ALPHA ** 2 * (nx + KAPPA) - nx
        # calculate weights
        Wm = [lamda / (lamda + nx)]
        Wc = [(lamda / (lamda + nx)) + (1 - ALPHA ** 2 + BETA)]
        for i in range(2 * nx):
            Wm.append(1.0 / (2 * (nx + lamda)))
            Wc.append(1.0 / (2 * (nx + lamda))) 
        
        gamma = np.sqrt(nx + lamda)
        Wm    = np.array(Wm)
        Wc    = np.array(Wc)
        
        ###############
        self.X     = X
        self.P     = P
        self.Q     = Q
        self.R     = R
        self.dt    = dt
        self.gamma = gamma
        self.Wm    = Wm
        self.Wc    = Wc 
        
    
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
    
    ###################
    def generate_sigma_points(self, x, P, gamma):
        sigma = x
        Psqrt = sqrtm(P)
        n     = len(x)
        
        # Positive direction
        for i in range(n):
            sigma = np.vstack((sigma, x + gamma * Psqrt[:, i]))
            
        # Negative direction
        for i in range(n):
            sigma = np.vstack((sigma, x - gamma * Psqrt[:, i]))
    
        return sigma.T
    
    ###################
    def predict(self):
        X       = self.X
        P       = self.P
        Q       = self.Q
        dt      = self.dt
        gamma   = self.gamma
        Wm      = self.Wm
        Wc      = self.Wc
        
        X_sgm   = self.generate_sigma_points(X, P, gamma)
        
        # # x=f(x) : Eular method
        # X_sgm   = np.array([X_sgm[:,i] + self.JansenRit_model(X_sgm[:6,i], X_sgm[6:,i]) * dt for i in range(len(Wc))]).T
        
        # x=f(x) : Runge-kutta method
        X_sgm   = np.array([self.state_func(X_sgm[:6,i], X_sgm[6:,i]) for i in range(len(Wc))]).T
        XPred   = (Wm @ X_sgm.T).T
        
        PPred   = np.zeros(P.shape) + Q
        for i in range(len(Wc)):
            d     = X_sgm[:,i] - XPred
            PPred = PPred + Wc[i] * d[:, np.newaxis] @ d[np.newaxis,:]
        
        self.X     = XPred
        self.P     = PPred
        self.X_sgm = X_sgm
    

    def update(self):
        z     = self.z
        X     = self.X
        P     = self.P
        
        R     = self.R
        dt    = self.dt
        gamma = self.gamma
        Wm    = self.Wm
        Wc    = self.Wc
        
        D     = np.array([
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                         ])
        ub    = np.array([10.00,  200,   60,  200,  315])
        lb    = np.array([ 0.01,   15, 0.01,   15,  110])
        b     = np.zeros(ub.shape)
        
        H     = np.array([[0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]])
        ################
        
        X_sgm = self.generate_sigma_points(X, P, gamma)
        z_sgm = H @ X_sgm
        zPred = (Wm @ z_sgm.T).T
        y     = z - zPred # prediction error of observation model
        
        Pxz   = np.zeros((len(X), len(y)))
        Pzz   = R #* np.random.randn()
        for i in range(len(Wc)):
            dx    = X_sgm[:,i] - X
            dz    = z_sgm[:,i] - zPred
            
            Pxz = Pxz + Wc[i] * dx[:, np.newaxis] @ dz[np.newaxis,:]
            Pzz = Pzz + Wc[i] * dz[:, np.newaxis] @ dz[np.newaxis,:]
        
        Pzz_inv = np.linalg.inv(Pzz)
        
        K       = Pxz @ Pzz_inv # Kalman Gain
        X_new   = X + K @ y        
        P_new   = P - K @ Pzz @ K.T
        
        
        ##### inequality constraints ##########################################
        ###   Constraint would be applied only when the inequality condition is not satisfied.
        I     = np.eye(len(X_new))
        W_inv = np.linalg.inv(P_new)
        L     = W_inv @ D.T @ np.linalg.inv(D @ W_inv @ D.T)
        value = D @ X_new 
        for i in range(len(value)):
            if (value[i] > ub[i]) | (value[i] < lb[i]):
                if (value[i] > ub[i]): 
                    b[i] = ub[i]
                elif (value[i] < lb[i]):
                    b[i] = lb[i]
        ## Calculate state variables with interval contraints
        X_c   = X_new - L @ (D @ X_new - b)
        
        for i in range(len(value)):
            if (value[i] > ub[i]) | (value[i] < lb[i]):
                X_new[i+6] = X_c[i+6]
        ##### inequality constraints ##########################################
        
            
        ### log-likelihood
        _, logdet = np.linalg.slogdet(Pzz)
        
        loglike   = -0.5 * (np.log(2*np.pi) + logdet + y @ Pzz_inv@ y)
        
        
        self.X       = X_new
        self.P       = P_new
        self.zPred   = zPred
        self.S       = Pzz
        self.R       = R
        self.loglike = loglike
    
    def ukf_estimation(self, z):   
        self.z = z
        # Prediction step (estimate state variable)
        self.predict()
        
        # Update state (Update parameters)
        self.update()
