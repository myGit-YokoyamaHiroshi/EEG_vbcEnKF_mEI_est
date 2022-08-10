#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:37:21 2021

@author: Hiroshi Yokoyama
"""
from IPython import get_ipython
from copy import deepcopy, copy
# get_ipython().magic('reset -sf')
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

if os.name != 'nt':
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/arial.ttf')
    matplotlib.rc('font', family="Arial")

plt.rcParams['font.family']      = 'Arial'
plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams['xtick.direction']  = 'in'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams["font.size"]        = 22 
plt.rcParams['lines.linewidth']  = 1.0
plt.rcParams['figure.dpi']       = 96
plt.rcParams['savefig.dpi']      = 600 
#%%
import sys
sys.path.append(current_path)

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from scipy import signal as sig
import numpy as np
import joblib
import multiprocessing as multi

def est_model(eeg, time, Nstate, Nt, Npar, dt):
    from my_modules.vb_enkf_JRmodel import vbEnKF_JansenRit
    
    # Estimation parameter of EnKF 
    A          = 3.25
    a          = 100
    B          = 22
    b          = 50
    p          = 220
    Q          = np.diag(np.hstack((1E-2 * dt * np.ones(6), 1E-3 * np.ones(5))))
    R          = 50
    
    xEst       = np.zeros(Nstate)
    PEst       = np.eye(Nstate)
    xEst[6:]   = np.array([A, a, B, b, p]) 
    
    #%%
    # SNRdB       = 10
    # sigpower    = np.mean(eeg**2);
    
    # noisepower  = sigpower/(10**(SNRdB/10));
    # noisesignal = np.random.randn(len(eeg))*np.sqrt(noisepower);

    x_pred    = np.zeros((Nt, Nstate))
    eeg_pred  = np.zeros(Nt)
    eeg_observe = eeg 
    x_pred[0,:] = xEst
    ELBO        = np.zeros(Nt)
    R_save      = np.zeros(Nt)
    R_save[0]   = R
    ## initialization
    model = vbEnKF_JansenRit(xEst, PEst, Q, R, dt, Npar)
    
    for t in range(1,Nt):
        z = eeg_observe[t-1] 
        ### update model
        model.vbenkf_estimation(z)
        
        # store data history
        PEst = model.P
        S    = model.S
        R    = (model.b/model.a) * model.R
        
        x_pred[t,:] = model.X
        eeg_pred[t] = model.zPred[0]
        ELBO[t]     = model.elbo
        R_save[t]   = R
        if np.mod(t+1, 10)==0:
            print('#itr.: %d (R = %5.8f)'%((t+1), R))
        #%%
    param_pred = x_pred[:,6:]
    
    #%%
    return eeg_pred, param_pred, R_save, Npar

def main():
    njob_max    = multi.cpu_count()
    #%% load synthetic data
    # fs_dwn      = 100
    Npar_list = np.arange(40, 520, 20)
    # Npar_list = np.arange(40, 60, 20)
    Ntri        = 50
    
    fullpath    = param_path + 'synthetic_data.npy'
    param_dict  = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
    eeg         = param_dict['eeg']
    t_true      = param_dict['t']
    fs          = param_dict['fs']
    dt          = param_dict['dt']
    
    x_true      = param_dict['y']     # exact value of satate variables 1 (numerical solution of Neural mass model)
    param_true  = param_dict['param'] # exact value of satate variables 2 (parameters of Neural mass model)
    
    eeg         = eeg + np.random.normal(loc = 0, scale=1.3, size=len(eeg))
    Nstate      = (x_true.shape[1]) + param_true.shape[1]
 
    
    # band        = fs_dwn/2  # Desired pass band, Hz
    # trans_width = 0.1   # Width of transition from pass band to stop band, Hz
    # numtaps     = 6000  # Size of the FIR filter.
    # b           = sig.firwin(numtaps, cutoff = band, fs=fs, width = trans_width, window = "hanning")
    # a           = 1
    # eeg         = sig.filtfilt(b, a, eeg).T
    # eeg         = sig.resample(eeg, int(len(eeg) * (fs_dwn/fs)))
    
    # Nt          = len(eeg)
    # dt          = 1/fs_dwn
    # time        = np.arange(0,Nt,1)/fs_dwn
    
    Nt          = len(eeg)
    dt          = 1/fs
    time        = np.arange(0,Nt,1)/fs
    
    #%%
    
    print(__file__ + " start!!")
    #%%
    
    for Npar in Npar_list:
        fig_save_dir = current_path + '/figures/Npar%03d/'%Npar 
        if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
            os.makedirs(fig_save_dir)
        
        np.random.seed(0)
        processed  = joblib.Parallel(n_jobs=int(0.8*njob_max), verbose=5)(joblib.delayed(est_model)(eeg, time, Nstate, Nt, Npar, dt) for i in range(Ntri))
        #%%
        eeg_pred   = np.array([processed[i][0] for i in range(len(processed))])
        param_pred = np.array([processed[i][1] for i in range(len(processed))])
        R_save     = np.array([processed[i][2] for i in range(len(processed))])
        #%%        
        plt.plot(time, eeg, label='exact', zorder=1);
        plt.plot(time, eeg_pred[0,:],    label='estimated', zorder=2);
        plt.xlabel('time (s)')
        plt.ylabel('amplitude (a.u.)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
        plt.savefig(fig_save_dir + 'eeg_Npar%d.png'%Npar, bbox_inches="tight")
        plt.savefig(fig_save_dir + 'eeg_Npar%d.svg'%Npar, bbox_inches="tight")
        plt.show()
        #%%
        plt.hist(R_save[:,-1])
        plt.xlabel('noise covariance')
        plt.ylabel('frequency')
        plt.xlim(0.5, 3)
        # plt.xlim(1.5, 3)
        plt.show()
        #%%    
        fig_name = ['A', 'a', 'B', 'b', 'p']
        ylims    = np.array([[  2,   7],
                             [ 80, 120],
                             [ 10,  30],
                             [ 40,  70],
                             [ 60, 420]])
        
        for i in range(len(fig_name)):
            fig, ax = plt.subplots()
            
            lines = [np.column_stack([time, param_pred[j,:,i]]) for j in range(Ntri)]
            
            lc = LineCollection(lines, cmap='Blues', linewidth=1)
            lc.set_array(np.arange(0, Ntri))
            lc.set_alpha(0.5)
            line = ax.add_collection(lc) #add to the subplot
            if fig_name[i] == 'p':
                plt.plot(t_true, param_true[:,i], c='k', linewidth=2, label='exact', zorder=0);
            else:
                plt.plot(t_true, param_true[:,i], c='k', linewidth=2, label='exact', zorder=Ntri);
            
            plt.xlabel('time (s)')
            plt.ylabel('amplitude (a.u.)')
            plt.ylim(ylims[i,:])
            plt.title('$' + fig_name[i] + '(t)$')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
            
            # add color bar below main figure panel
            divider = make_axes_locatable(ax)
            cax = divider.new_vertical(size='5%', pad=1.0, pack_start = True)
            fig.add_axes(cax)
            fig.colorbar(line, cax = cax, label='Num. of trial',  orientation='horizontal')
            
            plt.savefig(fig_save_dir + 'param_' + str(i+1) +'_Npar%d.png'%Npar, bbox_inches="tight")
            plt.savefig(fig_save_dir + 'param_' + str(i+1) +'_Npar%d.svg'%Npar, bbox_inches="tight")
            plt.show()
        #%%
        save_dict = {}
        save_dict['models']      = processed
        save_dict['eeg_observe'] = eeg
        save_dict['eeg_pred']    = eeg_pred
        save_dict['param_pred']  = param_pred
        save_dict['time']        = time # unit: s
        save_dict['dt']          = dt
        save_dict['fs']          = 1/dt
        save_dict['t_true']      = t_true
        save_dict['param_true']  = param_true
        save_dict['R_est']       = R_save
        
        
        save_name     = './save_data/model_est_Npar%03d'%Npar  
        fullpath_save = save_name 
        np.save(fullpath_save, save_dict)
#%%
if __name__ == '__main__':
    main()

    