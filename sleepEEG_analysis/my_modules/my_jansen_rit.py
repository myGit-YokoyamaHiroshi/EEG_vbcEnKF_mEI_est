# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:54:06 2020

@author: yokoyama
"""

from copy import deepcopy
from numpy.matlib import repmat
from numpy.random import randn, rand
import numpy as np

#%%
##############################################################################

def Sigm(v):
    v0   = 6
    vmax = 5
    r    = 0.56
    sigm = vmax / (1 + np.exp(r * ( v0 - v )))
    
    return sigm

def Sigm_diff(v):
    r    = 0.56
    vmax = 5
    sigm_diff = r/vmax * Sigm(v) * (vmax - Sigm(v))
    
    return sigm_diff

def postsynaptic_potential_function(y, z, A, a, Sgm):
    dy = z
    dz = A * a * Sgm - 2 * a * z - a**2 * y
    
    f_out = np.hstack((dy, dz))
    return f_out


def Jacobian(x, A, a, B, b, u):
    C     = 135
    c1    = C
    c2    = 0.8  * C
    c3    = 0.25 * C
    c4    = 0.25 * C
    
    #### Calculate Jacobian Jx = df/dx
    O     = np.zeros((3,3))
    I     = np.eye(3)
    Gamma = np.diag([a, a, b])
    dGdx  = np.array([
                      [                           0, A*a*Sigm_diff(x[1]-x[2]),  -A*a*Sigm_diff(x[1]-x[2])],
                      [A*a*c1*c2*Sigm_diff(c1*x[0]),                        0,                          0],
                      [B*b*c3*c4*Sigm_diff(c3*x[0]),                        0,                          0]
                      ])
    
    term1 = np.hstack((O, I))
    term2 = np.hstack((-Gamma**2 + dGdx, -2*Gamma))
    
    J_x = np.vstack((term1,term2))
    
    #### Calculate Jacobian Jpar = df/d par
    term3 = np.zeros((3, 5))
    term4 = np.array([
                      [     a*Sigm(x[1]-x[2]),      A*Sigm(x[1]-x[2])-2*x[3]-2*a*x[0],                   0,                                   0,   0],
                      [a*(u+c2*Sigm(c1*x[0])), A*(u+c2*Sigm(c1*x[0]))-2*x[4]-2*a*x[1],                   0,                                   0, A*a],
                      [                     0,                                      0,  b*c4*Sigm(c3*x[0]),  B*c4*Sigm(c3*x[0])-2*x[5]-2*b*x[2],   0]
                      ])
    J_par = np.vstack((term3, term4))
    # J_par = np.zeros((6,5))
    ##### combine two Jacobian matrix
    J     = np.hstack((J_x, J_par))
    return J

def func_JR_model(y, A, a, B, b, u):
    dy   = np.zeros(len(y))
    C    = 135
    c1   = 1.0  * C
    c2   = 0.8  * C
    c3   = 0.25 * C
    c4   = 0.25 * C
    
    Sgm_12 = Sigm(y[1] - y[2]);
    Sgm_p0 = u + c2 * Sigm(c1*y[0]);
    Sgm_0  = c4 * Sigm(c3*y[0]);
        
    dy_03 = postsynaptic_potential_function(y[0], y[3], A, a, Sgm_12);
    dy_14 = postsynaptic_potential_function(y[1], y[4], A, a, Sgm_p0);
    dy_25 = postsynaptic_potential_function(y[2], y[5], B, b, Sgm_0);
    
    # sort order of dy
    dy[0] = dy_03[0]
    dy[3] = dy_03[1]
    
    dy[1] = dy_14[0]
    dy[4] = dy_14[1]
    
    dy[2] = dy_25[0]
    dy[5] = dy_25[1]
    
    return dy

def func_JR_model_jacobian(y, A, a, B, b, u):
    par = np.array([A, a, B, b, u])
    J   = Jacobian(y, A, a, B, b, u)
    Y   = np.hstack((y, par))
    
    dY  = J @ Y
    
    return dY

def runge_kutta(dt, func, X_now, A, a, B, b, u):
    k1   = func(X_now, A, a, B, b, u)
    
    X_k2 = X_now + (dt/2)*k1
    k2   = func(X_k2, A, a, B, b, u)
    
    X_k3 = X_now + (dt/2)*k2
    k3   = func(X_k3, A, a, B, b, u)
    
    X_k4 = X_now + dt*k3
    k4   = func(X_k4, A, a, B, b, u)
    
    X_next = X_now + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return X_next

def euler_maruyama(dt, func, y, A, a, B, b, u, noise_scale):
    p      = noise_scale
    dw     = np.random.randn(y.shape[0])
    
    y_next = y + func(y, A, a, B, b, u) * dt + np.sqrt(dt) * p * dw
    
    return y_next
#%%
