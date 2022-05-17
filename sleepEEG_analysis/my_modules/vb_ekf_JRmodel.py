#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:49:09 2021

@author: user
"""

import numpy as np
from copy import deepcopy
from scipy.linalg import expm

class vbEKF_JansenRit:
    def __init__(self, X, P, Q, R, dt, eta):
        self.X  = X
        self.P  = P
        self.Q  = Q
        self.R  = R
        self.dt = dt
        self.a0    = 1E-4
        self.b0    = 1E-6
        self.eta   = eta # forgetting factor
        
        
        self.a     = self.a0 
        self.b     = self.b0
        
    
    def Sigm(self, v):
        v0   = 6
        vmax = 5
        r    = 0.56
        sigm = vmax / (1 + np.exp(r * ( v0 - v )))
        
        return sigm
    
    def Sigm_diff(self, v):
        r    = 0.56
        vmax = 5
        sigm_diff = r/vmax * self.Sigm(v) * (vmax - self.Sigm(v))
        
        return sigm_diff
    
    def Jacobian(self, x, par, dt):
        c1    = 135
        c2    = 0.8  * c1
        c3    = 0.25 * c1
        c4    = 0.25 * c1
        
        A       = par[0]
        a       = par[1]
        B       = par[2]
        b       = par[3]
        u       = par[4]
        
        #### Calculate Jacobian Jx = df/dx
        O     = np.zeros((3,3))
        I     = np.eye(3)
        diag  = np.diag([-2*a, -2*a, -2*b])
        dGdx  = np.array([
                          [                             -a*a, A*a*self.Sigm_diff(x[1]-x[2]),  -A*a*self.Sigm_diff(x[1]-x[2])],
                          [A*a*c1*c2*self.Sigm_diff(c1*x[0]),                          -a*a,                               0],
                          [B*b*c3*c4*self.Sigm_diff(c3*x[0]),                             0,                            -b*b]
                          ])
        
        term1 = np.hstack((O, I))
        term2 = np.hstack((dGdx, diag))
        
        J_x = np.vstack((term1,term2))
        #### Calculate Jacobian Jpar = df/d par
        term3 = np.zeros((3, len(par)))
        term4 = np.array([
                          [     a*self.Sigm(x[1]-x[2]),      A*self.Sigm(x[1]-x[2])-2*x[3]-2*a*x[0],                        0,                                        0,   0],
                          [a*(u+c2*self.Sigm(c1*x[0])), A*(u+c2*self.Sigm(c1*x[0]))-2*x[4]-2*a*x[1],                        0,                                        0, A*a],
                          [                          0,                                           0,  b*c4*self.Sigm(c3*x[0]),  B*c4*self.Sigm(c3*x[0])-2*x[5]-2*b*x[2],   0]
                          ])
        J_par = np.vstack((term3, term4))
        ##### combine two Jacobian matrix
        J     = np.hstack((J_x, J_par))
        return J
    
    ###################
    def predict(self):
        X       = self.X
        P       = self.P
        Q       = self.Q
        dt      = self.dt
        
        x       = deepcopy(X[:6])
        par     = deepcopy(X[6:])
        
        Npar    = len(par)
        Nx      = len(x)
        
        ### Calculate Jacobian matrix A
        J       = self.Jacobian(x, par, dt)
        tmp     = np.zeros((Npar, Nx + Npar))#np.hstack((np.zeros((Npar,Nx)), np.eye(Npar)))
        A       = np.vstack((J, tmp)) 
        
        ### Convert from Jacobian matrix A to State transition matrix F
        # Calculate Matrix exponential with 1st oder Taylor approximation 
        # F       = np.eye(len(X)) + A*dt # F = exp (A * dt) = I + A * dt
        # Calculate Matrix exponential with scipy module
        F       = expm(A * dt)#  F = exp (A * dt)
        ## Update State model
        X_new   = F @ X #  F X = exp (A * dt) X
        
        x_new   = X_new[:6]#x + X_new[:6] * dt + eta * np.sqrt(dt) * np.random.normal(loc=0, scale=1, size=Nx)
        par_new = X_new[6:]
        
        XPred = np.hstack((x_new, par_new))
        PPred = F @ P @ F.T + Q
        
        self.X = XPred
        self.P = PPred
    

    def update(self):
        z     = self.z
        X     = self.X
        P     = self.P
        R     = self.R
        dt    = self.dt
        a       = self.a
        b       = self.b

        prec_param = a/b
        
        D     = np.array([
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                         ])
        ub    = np.array([100.00,  200, 100.00,  200, 320])
        lb    = np.array([  0.01,    5,   0.01,    5, 120])
        b     = np.zeros(ub.shape)
        
        H     = np.array([[0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]])
        zPred = H @ X
        y     = z - zPred # prediction error of observation model
        S     = H @ P @ H.T + (1/prec_param) * R
        
        S_inv = np.linalg.inv(S)
            
        K     = P @ H.T @ S_inv
        X_new = X + K @ y
        P_new = P - K @ H @ P # P - K @ S @ K.T
        
        ##### inequality constraints ##########################################
        ###   Constraint would be applied only when the inequality condition is not satisfied.
        W_inv = np.linalg.inv(P_new)
        L     = W_inv @ D.T @ np.linalg.pinv(D @ W_inv @ D.T)
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
            
        
        #################### update Normal-Gamma parameter
        a         = self.a + len(y)/2
        b         = self.b + 1/2 * np.sum((z - H@X_new)**2) + 1/2 * np.trace(H@P_new@H.T) 
        
        ### log-likelihood
        _, logdet = np.linalg.slogdet(S)
        
        loglike   = -0.5 * (np.log(2*np.pi) + logdet + y @ S_inv@ y)
        
        
        self.X       = X_new
        self.P       = P_new
        self.zPred   = zPred
        self.S       = S
        self.R       = R
        self.loglike = loglike
        self.a       = self.eta * a
        self.b       = self.eta * b
    
    def vbekf_estimation(self, z):   
        self.z = z
        # Prediction step (estimate state variable)
        self.predict()
        
        # Update state (Update parameters)
        self.update()
