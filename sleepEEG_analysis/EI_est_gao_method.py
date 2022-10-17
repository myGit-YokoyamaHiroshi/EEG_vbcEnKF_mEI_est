#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 16:13:47 2021

@author: Hiroshi Yokoyama
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
from matplotlib import font_manager
import matplotlib
if os.name == 'posix': # for linux
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/arial.ttf')
    matplotlib.rc('font', family="Arial")

plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 26 # Font size
#%%
import sys
sys.path.append(current_path)

from mne.time_frequency import tfr_array_morlet as timef_wt
from my_modules.preprocessing import Preprocessing
from scipy import signal as sig
from scipy.optimize import least_squares
import joblib
import numpy as np
from joblib import Parallel, delayed
import multiprocessing as multi
import pyedflib
#%%
def fun(x, t, y):
    return (x[0] + x[1] * np.log10(t)) - y

def calc_EIslope(t, power, freqs):
    res_robust = least_squares(fun, np.ones(2), loss='soft_l1', f_scale=0.1, args=(freqs, power[:,t]))
    coef_all   = res_robust.x 
    EI_slope   = res_robust.x[1]
    
    return coef_all, EI_slope, t
############################################################################## 
#%% check the directory and get the file name automatically
name     = []
ext      = []
# file_dir = current_path + '/raw_data/'
file_dir = './raw_data/' 
for file in os.listdir(file_dir):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
    
    print(split_str)
print('---------------------')

fullpath   = './save_data/subject_list.npy'
load_dict  = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
list_sbj   = load_dict['list_subject']

del load_dict
#%%
fig_save_dir = current_path + '/figures/' 
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

data_save_dir = current_path + '/save_data/gao_method/' 
if os.path.exists(data_save_dir)==False:  # Make the directory for data saving
    os.makedirs(data_save_dir)
#%%
for sbjID in list_sbj:
    #%% load data
    np.random.seed(0)
    
    f_edf   = [s for s in name if ('PSG' in s) and (sbjID in s)][0]  + '.edf'
    dat_edf = pyedflib.EdfReader(file_dir+ f_edf)
    Headers = dat_edf.getSignalHeaders()
    Nch     = len(Headers)
    
    for ch in range(Nch):
        label = Headers[ch]['label']
        if ('Pz-Oz' in label): #   ('Fpz-Cz' in label):#
            idx = ch
            break
        
    fs      = dat_edf.getSampleFrequencies()[idx]
    raw_sig = dat_edf.readSignal(idx)
    Nt      = len(raw_sig)
    time    = np.arange(1, Nt+1)/fs
    
    del Headers
    del dat_edf
    
    f_anno    = [s for s in name if ('Hypnogram' in s) and (sbjID in s)][0]  + '.edf'
    anno_edf  = pyedflib.EdfReader(file_dir+ f_anno)
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
        
        if (stage_lab[i][-1]=='W') or (stage_lab[i][-1]=='?'):
            state_all[tmp_idx] = 1
        elif stage_lab[i][-1]=='R':
            state_all[tmp_idx] = 0
        else:
            if stage_lab[i][-1]=='4':
                state_all[tmp_idx] = -3
            else:
                state_all[tmp_idx] = -int(stage_lab[i][-1])  
    #%%
    baseline    = 60*20
    if (stage_dur[0]>=3600) and (stage_lab[0]=='Sleep stage W'):
        st_idx = state_interval[1,0]
    else:
        st_idx = state_interval[0,0]
    
    end_idx     = state_interval[REM_idx[0], 1]
    target_idx  = np.arange(st_idx-int(fs*baseline), end_idx+1, step=1, dtype=int)
    
    REM_st      = int(state_interval[REM_idx[0], 0]- st_idx + int(fs*baseline))
    #%% preprocessing
    prepro      = Preprocessing(raw_sig[target_idx,np.newaxis,np.newaxis], np.arange(0, len(target_idx)), np.array([]), np.array(['']), fs, 25)
    filt_sig    = prepro.denoising()[:,0,0]

    # band        = [.6, 49.9]  # Desired pass band, Hz
    # trans_width = 0.1   # Width of transition from pass band to stop band, Hz
    # numtaps     = 6000  # Size of the FIR filter.
    # b           = sig.firwin(numtaps, cutoff = band, fs=fs, width = trans_width, window = "hanning", pass_zero = False)
    # a           = 1
    
    # filt_sig    = sig.filtfilt(b, a, raw_sig[target_idx] -  raw_sig[target_idx].mean()).T
    
    state       = state_all[target_idx]
    time        = time[target_idx] - time[int(st_idx)]
    eeg_observe = filt_sig
    
    del raw_sig
    #%%
    freqs      = np.arange(30, 50.5, .5)
    tmp_tf = timef_wt(epoch_data = eeg_observe[np.newaxis,np.newaxis,:], sfreq = 100, freqs = freqs, 
                      n_cycles=16, zero_mean=False, use_fft=True, 
                      decim=1, output='complex', verbose=None)
    power  = np.real(np.log10(abs(tmp_tf)**2)[0,0,:,:])
    #%%    
    # coeff, EIslope = calc_EIslope(power, np.arange(30, 51, 1))    
    Nfrqs, Nt  = power.shape
    processed  = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(calc_EIslope)(t, power, freqs) for t in range(Nt))
    processed.sort(key=lambda x: x[2]) # sort the output list according to the model order
    
    coeff      = np.array([tmp[0] for tmp in processed])
    EIslope    = np.array([tmp[1] for tmp in processed])
    ##########################################################################
    #%%
    fig = plt.figure(figsize=(8, 10))
    gs  = fig.add_gridspec(3, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.1)
    
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(time/60, EIslope)
    ax1.plot([0,0], [-300, 300], 'r', linestyle='--', linewidth=4, zorder=0)
    ax1.axvspan(time[REM_st]/(60), time[-1]/(60), color='gray', alpha=0.3)
    ax1.set_xlim(time[0]/60, time[-1]/60)
    ax1.set_ylim(-40, 30)
    # plt.xlim(-21, time[-1]/60)
    ax1.set_xticklabels([])
    ax1.set_ylabel('mean E/I ratio (a.u.)')
    plt.grid()
    
    ###########
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(time/60, state, 'k', linestyle='-', linewidth=2)
    ax2.plot([0,0], [-4, 2], 'r', linestyle='--', linewidth=4, zorder=0)
    ax2.axvspan(time[REM_st]/(60), time[-1]/(60), color='gray', alpha=0.3)
    ax2.set_yticks([-3, -2, -1, 0, 1], )
    ax2.set_yticklabels(['Stage 3/4', 'Stage 2', 'Stage 1', 'REM', 'Wake'])
    ax2.set_xlabel('time (min)')
    ax2.set_xlim(time[0]/60, time[-1]/60)
    ax2.set_ylim(-3.5, 1.5)
    plt.grid()
    ########
    # figname = fig_save_dir + sbjID + '_EI_ratio'
    # plt.savefig(figname + '.png', bbox_inches="tight")
    # plt.savefig(figname + '.svg', bbox_inches="tight")
    
    plt.show()
    #%% Save data
    save_dict = {}
    save_dict['eeg_observe'] = eeg_observe
    save_dict['sleep_state'] = state
    save_dict['time']        = time # unit: s
    save_dict['idx_REM_st']  = REM_st
    save_dict['t_REM_st']    = time[REM_st] # unit: s
    save_dict['EIslope']     = EIslope
    save_dict['fs']          = fs
    save_dict['coeff']       = coeff 
    
    save_name   = sbjID + 'EI_est_gao_method'  
    fullpath_save   = data_save_dir + save_name 
    np.save(fullpath_save, save_dict)
    
    del save_dict
    del eeg_observe, coeff, EIslope
    