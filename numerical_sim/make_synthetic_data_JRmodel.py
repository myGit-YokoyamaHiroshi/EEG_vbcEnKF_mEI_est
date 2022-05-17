#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 14:53:08 2021

@author: Hiroshi Yokoyama
"""
from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
current_path = os.path.dirname(__file__)
os.chdir(current_path)

fig_save_dir = current_path + '/figures/' 
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

current_path = os.getcwd()
param_path   = current_path + '/save_data/' 
if os.path.exists(param_path)==False:  # Make the directory for figures
    os.makedirs(param_path)

    
import matplotlib.pylab as plt
from matplotlib import font_manager
import matplotlib
font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/arial.ttf')
matplotlib.rc('font', family="Arial")

plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 26 # Font size


#%%
import sys
sys.path.append(current_path)

from my_modules.my_jansen_rit import *
from scipy import signal as sig
import numpy as np
import joblib
import random

np.random.seed(0)
#%%
def sigmoid(A, k,x,x0,b):
    return (A / (1 + np.exp(-k*(x-x0)))) + b

#%%
fs          = 5000
dt          = 1/fs
Nt          = int(20*fs)# + 100
t           = np.arange(0,Nt,1)/fs



y           = np.zeros((Nt, 6))
y_init      = y[0,:]
dy          = np.zeros((Nt, 6))
param       = np.zeros((Nt, 4))

# A           = sigmoid(5.5-3.25, 1, t, 15, 3.25) 
# a           = 100  * np.ones(Nt)
# B           = sigmoid(22-19, -1, t, 15, 19) 
# b           = 50   * np.ones(Nt)
# p           = np.random.normal(loc=  220, scale= 22, size=Nt)

A             = 3.25 * np.ones(Nt)
A[int(Nt/2):] = 4.25
a             = 100  * np.ones(Nt)
B             = 22   * np.ones(Nt) 
B[int(Nt/2):] = 19
b             = 50   * np.ones(Nt)
b[int(Nt/2):] = 52
p             = np.random.normal(loc=  220, scale= 22, size=Nt)

dy[0, :]    = func_JR_model(y_init, A[0], a[0], B[0], b[0], p[0])

for i in range(1, Nt):        
    y_now      = y[i-1, :]
    y_next     = runge_kutta(dt, func_JR_model, y_now, A[i], a[i], B[i], b[i], p[i])
    dy[i, :]   = func_JR_model(y_now, A[i], a[i], B[i], b[i], p[i])
    y[i, :]    = y_next
    
eeg   = y[:,1]-y[:,2]
param = np.concatenate((A[:,np.newaxis], a[:,np.newaxis], B[:,np.newaxis], b[:,np.newaxis], p[:,np.newaxis]), axis=1)

#%%

fig = plt.figure(figsize=(10, 20))
gs  = fig.add_gridspec(4,2)
plt.subplots_adjust(wspace=0.5, hspace=0.6)

ax = fig.add_subplot(gs[0, 0:])
ax.plot(t, eeg)
ax.set_xlabel('time (s)')
ax.set_ylabel('amplitude (a.u.)')
ax.set_xticks(np.arange(0, 30, 10))
ax.set_title('synthetic EEG')

ax = fig.add_subplot(gs[1, 0])
ax.plot(t, A)
ax.set_xlabel('time (s)')
ax.set_ylabel('amplitude (a.u.)')
ax.set_title('$A(t)$')
ax.set_xticks(np.arange(0, 30, 10))
ax.set_ylim(2.25, 7.25)

ax = fig.add_subplot(gs[1, 1])
ax.plot(t, B)
ax.set_xlabel('time (s)')
# ax.set_ylabel('amplitude (a.u.)')
ax.set_title('$B(t)$')
ax.set_xticks(np.arange(0, 30, 10))
ax.set_ylim(18, 23)

ax = fig.add_subplot(gs[2, 0])
ax.plot(t, a)
ax.set_xlabel('time (s)')
ax.set_ylabel('amplitude (a.u.)')
ax.set_title('$a(t)$')
ax.set_xticks(np.arange(0, 30, 10))
ax.set_ylim(98, 102)

ax = fig.add_subplot(gs[2, 1])
ax.plot(t, b)
ax.set_xlabel('time (s)')
# ax.set_ylabel('amplitude (a.u.)')
ax.set_title('$b(t)$')
ax.set_xticks(np.arange(0, 30, 10))
ax.set_ylim(48, 54)

ax = fig.add_subplot(gs[3, 0])
ax.plot(t, p)
ax.set_xlabel('time (s)')
ax.set_ylabel('amplitude (a.u.)')
ax.set_title('$p(t)$')
ax.set_xticks(np.arange(0, 30, 10))
ax.set_ylim(60, 420)

fig_save_dir = current_path + '/figures/'
plt.savefig(fig_save_dir + 'synthetic_data.png', bbox_inches="tight")
plt.savefig(fig_save_dir + 'synthetic_data.svg', bbox_inches="tight")
plt.show()
#%%
frqs, p_welch = sig.welch(eeg, fs, nperseg = Nt/2)
plt.plot(frqs, 10*np.log10(p_welch));
plt.xlim(0, 60)
plt.xlabel('frequency (Hz)')
plt.ylabel('power (dB)')
plt.show()

#%%
############# save_data
param_dict          = {} 
param_dict['fs']    = fs
param_dict['dt']    = dt
param_dict['Nt']    = Nt#-100
param_dict['param'] = param#[100:,:]
param_dict['eeg']   = eeg#[100:]
param_dict['t']     = t#[100:]
param_dict['dy']    = dy
param_dict['y']     = y

save_name   = 'synthetic_data'
fullpath_save   = param_path + save_name 
np.save(fullpath_save, param_dict)