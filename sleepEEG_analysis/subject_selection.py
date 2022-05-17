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
param_path   = current_path + '/save_data/' 
if os.path.exists(param_path)==False:  # Make the directory for figures
    os.makedirs(param_path)
    
import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 26 # Font size
#%%
import sys
sys.path.append(current_path)

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
############################################################################## 
#%% check the directory and get the file name automatically
name     = []
ext      = []
file_dir = current_path + '/save_data/' 
for file in os.listdir(file_dir):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
    
    print(split_str)
print('---------------------')
#%%
fig_save_dir = current_path + '/figures/' 
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)
#%%

list_subject = []
for fname, extention in zip(name, ext):
    #%%
    if (extention == '.edf') and ('PSG' in fname):
        dat_edf = pyedflib.EdfReader(file_dir+ fname + extention)
        Headers = dat_edf.getSignalHeaders()
        Nch     = len(Headers)
        
        for ch in range(Nch):
            label = Headers[ch]['label']
            if ('Pz-Oz' in label): # ('Fpz-Cz' in label):# 
                idx = ch
                break
        
        
        fs      = dat_edf.getSampleFrequencies()[idx]
        raw_sig = dat_edf.readSignal(idx)
        Nt      = len(raw_sig)
        time    = np.arange(1, Nt+1)/fs
        
        del Headers
        del dat_edf
        
        IDs    = fname.split('-')[0][:-2]
        f_hyp  = [s for s in name if ('Hypnogram' in s) and (IDs in s)][0]
        
  
        anno_edf  = pyedflib.EdfReader(file_dir+ f_hyp + extention)
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
        band        = [1, 30]  # Desired pass band, Hz
        trans_width = .5    # Width of transition from pass band to stop band, Hz
        numtaps     = 600       # Size of the FIR filter.
        edges       = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
        b           = sig.remez(numtaps, edges, [0, 1, 0], Hz=fs)
        a           = 1
        
        filt_sig    = sig.filtfilt(b, a, raw_sig -  raw_sig.mean()).T
        
        state       = state_all[target_idx]
        filt_sig    = filt_sig[target_idx]
        time        = time[target_idx] - time[int(st_idx)]
        raw_sig     = raw_sig[target_idx]
        #%%
        
        fig = plt.figure(figsize=(8, 10))
        gs  = fig.add_gridspec(3, 1)
        plt.subplots_adjust(wspace=0.4, hspace=0)
        
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.plot(time, raw_sig);
        ax1.axvspan(time[0], time[int(fs*baseline)], color='gray', alpha=0.3)
        ax1.axvspan(time[REM_st], time[-1], color='gray', alpha=0.3)
        ax1.set_xticklabels([])
        ax1.set_ylabel('amplitude ($\mu V$)')
        
        #########
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.plot(time/60, state)
        ax2.axvspan(time[0]/(60), time[int(fs*baseline)]/(60), color='gray', alpha=0.3)
        ax2.axvspan(time[REM_st]/(60), time[-1]/(60), color='gray', alpha=0.3)
        ax2.set_yticks([-3, -2, -1, 0, 1])
        ax2.set_yticklabels(['Stage 3/4', 'Stage 2', 'Stage 1', 'REM', 'Wake'])
        ax2.set_xlabel('time (min)')
        
        ax1.set_title(IDs)
        plt.grid()
        
        figname = fig_save_dir + IDs + '_sleep_stage'
        plt.savefig(figname + '.png', bbox_inches="tight")
        plt.savefig(figname + '.svg', bbox_inches="tight")
        
        plt.show()
        
        
        
        num_wake = np.sum(state[int(fs*baseline)+1:]==1)
        
        if num_wake == 0:
            print(IDs)
            list_subject.append(IDs)
        #%%
save_dict = {}
save_dict['list_subject'] = list_subject 

save_name   = 'subject_list' 
fullpath_save   = current_path + '/save_data/' + save_name 
np.save(fullpath_save, save_dict)
