#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:13:47 2021

@author: user
"""
fdir = __file__ 
from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os

current_path = os.path.dirname(__file__)
os.chdir(current_path)



current_path = os.getcwd()
param_path   = '../save_data/' 
if os.path.exists(param_path)==False:  # Make the directory for figures
    os.makedirs(param_path)
    
import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 26 # Font size
#%%
import sys
sys.path.append(current_path)

# from my_modules.ekf_JRmodel import EKF_JansenRit
# from my_modules.ukf_JRmodel import UKF_JansenRit
# from my_modules.vb_ekf_JRmodel import vbEKF_JansenRit
# from my_modules.vb_ukf_JRmodel import vbUKF_JansenRit
from my_modules.vb_enkf_JRmodel import vbEnKF_JansenRit
from my_modules.preprocessing import Preprocessing
from scipy import signal as sig
import scipy.linalg
import math

import numpy as np
import joblib
import random

import pyedflib
#%%
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def epoch_average(time, value, win):
    t_epc    = np.arange(time[0], time[-1], win)
    interval = np.concatenate((t_epc[:-1, np.newaxis], t_epc[1:, np.newaxis]), axis=1)
    Nepc     = interval.shape[0]
    
    ave_epc  = np.zeros(t_epc.shape)
    sd_epc   = np.zeros(t_epc.shape)
    for epc in range(Nepc):
        idx          = np.where((time>=interval[epc,0]) & (time<=interval[epc,1]))[0]
        ave_epc[epc] = np.nanmean(value[idx])
        sd_epc[epc]  = np.nanstd(value[idx])
    
    return t_epc, ave_epc, sd_epc

def plot_response(fs, w, h, title):
    "Utility function to plot response functions"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
    ax.set_ylim(-50, 5)
    ax.set_xlim(20, 60)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)
    
    return fig, ax
############################################################################## 

#%%
fig_save_dir = current_path + '/figures/' 
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)
#%%
np.random.seed(0)

file_dir = '../raw_data/'
# fname1   = 'SC4091E0-PSG.edf'
# fname2   = 'SC4091EC-Hypnogram.edf'
# fname1   = 'SC4032E0-PSG.edf'
# fname2   = 'SC4032EP-Hypnogram.edf'
fname1   = 'SC4071E0-PSG.edf'
fname2   = 'SC4071EC-Hypnogram.edf'

# fname1   = 'SC4022E0-PSG.edf'
# fname2   = 'SC4022EJ-Hypnogram.edf'


# fname1   = 'SC4522E0-PSG.edf'
# fname2   = 'SC4522EM-Hypnogram.edf'

# fname1   = 'SC4401E0-PSG.edf'
# fname2   = 'SC4401EC-Hypnogram.edf'


dat_edf = pyedflib.EdfReader(file_dir+ fname1)
Headers = dat_edf.getSignalHeaders()
Nch     = len(Headers)

for ch in range(Nch):
    label = Headers[ch]['label']
    if ('Fpz-Cz' in label):#  ('Pz-Oz' in label): # 
        idx = ch
        break


fs       = dat_edf.getSampleFrequencies()[idx]
raw_sig  = dat_edf.readSignal(idx)
Nt       = len(raw_sig)
time_raw = np.arange(1, Nt+1)/fs

del Headers
del dat_edf
  
anno_edf  = pyedflib.EdfReader(file_dir+ fname2)
anno      = anno_edf.readAnnotations()

stage_idx = anno[0]*fs
stage_dur = anno[1]*fs
stage_lab = anno[2]

del anno 
del anno_edf

#%% extract target epoch
REM_idx        = np.where(stage_lab=='Sleep stage R')[0]
state_all      = np.zeros(raw_sig.shape)
state_interval = np.zeros((REM_idx[0]+1, 2), dtype=int) 

for i in range(REM_idx[0]+1):
    tmp_idx = np.arange(stage_idx[i], stage_idx[i+1], dtype=int)
    state_interval[i, :] = [stage_idx[i], stage_idx[i+1]]
    
    if (stage_lab[i][-1]=='W') or (stage_lab[i][-1]=='?') or (stage_lab[i]=='Movement time'):
        state_all[tmp_idx] = 1
    elif stage_lab[i][-1]=='R':
        state_all[tmp_idx] = 0
    else:
        if stage_lab[i][-1]=='4':
            state_all[tmp_idx] = -3
        else:
            state_all[tmp_idx] = -int(stage_lab[i][-1])  

#%%
baseline    = 60*30

if (stage_dur[0]>=3600) and (stage_lab[0]=='Sleep stage W'):
    st_idx = state_interval[1,0]
else:
    st_idx = state_interval[0,0]

end_idx     = state_interval[REM_idx[0], 1]
target_idx  = np.arange(st_idx-int(fs*baseline), end_idx+1, step=1, dtype=int)

REM_st      = int(state_interval[REM_idx[0], 0]- st_idx + int(fs*baseline))

#%% preprocessing
# band        = [1, 20]  # Desired pass band, Hz
# trans_width = .1   # Width of transition from pass band to stop band, Hz
# numtaps     = 6000  # Size of the FIR filter.
# b           = sig.firwin(numtaps, cutoff = band, fs=fs, width = trans_width, window = "hanning", pass_zero = False)
# a           = 1

# filt_sig    = sig.filtfilt(b, a, raw_sig -  raw_sig.mean()).T
# filt_sig    = filt_sig[target_idx]

band        = [1, 49] 
prepro      = Preprocessing(raw_sig[target_idx,np.newaxis,np.newaxis], np.arange(0, len(target_idx)), np.array([]), np.array(['Pz-Oz']), fs, np.array(band), 24)
filt_sig    = prepro.denoising()[:,0,0] 

f,  Pxx  = sig.welch(filt_sig, fs, nfft=4048, nperseg=4048)
f_, Pxx_ = sig.welch(raw_sig[target_idx], fs, nfft=4048, nperseg=4048)

plt.plot(f_, 10*np.log10(Pxx_))
plt.plot(f , 10*np.log10(Pxx))
plt.xlabel('frequency (Hz)')
plt.ylabel('power (dB)')
plt.show()
#%%
state       = state_all[target_idx]
time        = time_raw[target_idx] - time_raw[int(st_idx)]
eeg_observe = filt_sig
# time_raw    = time_raw[target_idx] - time_raw[int(st_idx)]
# fs_dwn      = 50
# eeg_observe = sig.decimate(filt_sig, int(fs/fs_dwn))
# time        = np.arange(0, len(eeg_observe))/fs_dwn - abs(time_raw[0])

plt.plot(time, raw_sig[target_idx]);
plt.plot(time, eeg_observe);
plt.axvspan(time[0], time[int(fs*baseline)], color='gray', alpha=0.3)
plt.axvspan(time[REM_st], time[-1], color='gray', alpha=0.3)
plt.show()
#%% Parameter initialization for UKF 
# dt          = 1/fs_dwn
dt          = 1/fs
Nstate      = 11
Nt          = len(time)
A           = 3.22
a           = 100
B           = 22
b           = 50
p           = 220

Q           = np.diag(np.hstack((1E-2 * np.ones(6), 1E-4 * np.ones(5))))
# Q           = np.diag(np.hstack((1E-2 * np.ones(6), 1E-8 * np.ones(5))))
R           = 1
eta         = 1


xEst        = np.zeros(Nstate)
xEst[6:]    = np.array([A, a, B, b, p])
PEst        = np.eye(Nstate)
#%% Apply UKF 
# history
x_pred      = np.zeros((Nt, Nstate))
eeg_pred    = np.zeros(Nt)

x_pred[0,:] = xEst
loglike     = np.zeros(Nt)
R_save      = np.zeros(Nt)
R_save[0]   = R
    
## initialization
# model = vbEKF_JansenRit(xEst, PEst, Q, R, dt, eta)
# model = vbUKF_JansenRit(xEst, PEst, Q, R, dt, eta)
model = vbEnKF_JansenRit(xEst, PEst, Q, R, dt, eta)

for t in range(1,Nt):
    z = eeg_observe[t-1]
    ### update model
    # model.vbekf_estimation(z)
    # model.vbukf_estimation(z)
    model.vbenkf_estimation(z)
    
    # store data history
    PEst = model.P
    S    = model.S
    R    = (model.b/model.a)*model.R
    
    x_pred[t,:] = model.X
    eeg_pred[t] = model.zPred[0]
    loglike[t]  = model.loglike
    R_save[t]   = R
    
    if np.mod(t+1, 50)==0:
        err = abs(z - model.zPred[0])**2
        print('#itr.: %d (R = %.4f, err. = %.4f)'%((t+1), R, err))
#%% Caluculate E/I ratio
win        = 3#unit: s
param_pred = x_pred[:,6:]
EIR        = param_pred[:,0]/(param_pred[:,0] + param_pred[:,2]) # = A/(A+B) 
base       = EIR[0:int(fs*baseline)].mean()
#%%
####### Calculate relative E/I changes
EI_change  = (EIR-base)/base * 100

t_epc, EI_change_epc, SD_epc = epoch_average(time/60, EI_change, win)

####### Plot result
fig = plt.figure(figsize=(8, 10))
gs  = fig.add_gridspec(3, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.1)

ax1 = fig.add_subplot(gs[0:2, 0])
# ax1.errorbar(t_epc, EI_change_epc, yerr=SD_epc, marker='o', mfc='black', ms=10, linestyle='-')
# ax1.plot(time/60, EI_change)
ax1.plot(time/60, EIR)
ax1.plot([0,0], [-300, 300], 'r', linestyle='--', linewidth=4, zorder=0)
# ax1.axvspan(time[0]/(60), time[int(fs*baseline)]/(60), color='gray', alpha=0.3)
ax1.axvspan(time[REM_st]/(60), time[-1]/(60), color='gray', alpha=0.3)
ax1.set_xlim(time[0]/60, time[-1]/60)
ax1.set_ylim(-0.05, 1.0)
# plt.xlim(-21, time[-1]/60)
ax1.set_xticklabels([])
ax1.set_ylabel('mean E/I ratio (a.u.)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.grid()
###########
ax2 = fig.add_subplot(gs[2, 0])
ax2.plot(time/60, state, 'k', linestyle='-', linewidth=2)
ax2.plot([0,0], [-4, 2], 'r', linestyle='--', linewidth=4, zorder=0)
# ax2.axvspan(time[0]/(60), time[int(fs*baseline)]/(60), color='gray', alpha=0.3)
ax2.axvspan(time[REM_st]/(60), time[-1]/(60), color='gray', alpha=0.3)
ax2.set_yticks([-3, -2, -1, 0, 1], )
ax2.set_yticklabels(['Stage 3/4', 'Stage 2', 'Stage 1', 'REM', 'Wake'])
ax2.set_xlabel('time (min)')
ax2.set_xlim(time[0]/60, time[-1]/60)#set_xlim(-3, time[-1]/60)
ax2.set_ylim(-3.5, 1.5)
plt.grid()
########

plt.show()
#%%
plt.plot(time/(60), eeg_pred,zorder=2, label='our model');
plt.plot(time/(60), eeg_observe,zorder=1, label='raw');
# plt.xlim(60, 60.5)
plt.ylim(-abs(eeg_observe).max(), abs(eeg_observe).max())
plt.xlabel('time (min)')
plt.ylabel('amplitude ($\mu V$)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.show()