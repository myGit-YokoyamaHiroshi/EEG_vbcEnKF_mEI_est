#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:37:21 2021

@author: Hiroshi Yokoyama
"""
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
from matplotlib.colors import ListedColormap, BoundaryNorm
from my_modules.vb_enkf_JRmodel import vbEnKF_JansenRit
import seaborn as sns
import numpy as np



def main():
    #%% load synthetic data
    fs_dwn    = 100
    Npar_list = np.arange(40, 520, 20)
    # Npar_list = np.arange(50, 550, 50)
    Ntri      = 50
    
        
    
    MAE       = np.zeros((Ntri, len(Npar_list))) 
    R_est_all = np.zeros((Ntri, len(Npar_list))) 
    
    #%%
    
    print(__file__ + " start!!")
    #%%
    np.random.seed(0)
    cnt = 0
    for Npar in Npar_list:
        save_name  = './save_data/model_est_Npar%03d'%Npar  
        fullpath   = save_name + '.npy'
        data_dict  = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
        processed  = data_dict['models']
        eeg        = data_dict['eeg_observe'] 
        eeg_pred   = data_dict['eeg_pred']
        param_pred = data_dict['param_pred']
        time       = data_dict['time']
        t_true     = data_dict['t_true']
        param_true = data_dict['param_true']
        R_est      = data_dict['R_est']
        
        R_est_all[:,cnt] = R_est[:,-1]
        MAE[:,cnt]       = np.mean(abs(eeg[:-1]-eeg_pred[:,1:]), axis=1)
        
        fig_save_dir = current_path + '/figures/Npar%03d/'%Npar 
        if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
            os.makedirs(fig_save_dir)
        
        #%%
        fig = plt.figure(figsize=(10, 6))
        gs  = fig.add_gridspec(2,3)
        plt.subplots_adjust(wspace=0.5, hspace=0.6)
        
        
        ax1 = fig.add_subplot(gs[0, 0:3])
        ax1.plot(time, eeg, label='exact', zorder=1);
        ax1.plot(time[1:], eeg_pred[0,:-1],    label='estimated', zorder=2);
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('amplitude (a.u.)')
        ax1.set_ylim(-1, 18)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
        ############
        time_range = np.array([[ 2,  4],
                               [ 9, 11],
                               [16, 18]])
        for i in range(3):
            axn = fig.add_subplot(gs[1, i])
            axn.plot(time, eeg, label='exact', zorder=1);
            axn.plot(time[:-1], eeg_pred[0,1:],    label='estimated', zorder=2);
            axn.set_xlabel('time (s)')
            axn.set_xlim(time_range[i,:])
            axn.set_ylim(-1, 18)
            if i == 0:
                axn.set_ylabel('amplitude (a.u.)')
        #######
        plt.savefig(fig_save_dir + 'eeg_Npar%d.png'%Npar, bbox_inches="tight")
        plt.savefig(fig_save_dir + 'eeg_Npar%d.svg'%Npar, bbox_inches="tight")
        plt.show()
        
        cnt += 1
    #%%
    fig_save_dir = current_path + '/figures/'
    
    tick_idx = np.where(np.mod(Npar_list, 100)==0)[0]
    
    xtick_lab = []
    for idx in tick_idx:
        xtick_lab.append(str(Npar_list[idx]))
    
    # plt.scatter(Npar_list, MAE.mean(axis=0))
    sns.violinplot(data=MAE, color='skyblue')
    plt.xticks(ticks=tick_idx, labels = xtick_lab)
    plt.xlabel('Num. of ensemble')
    plt.ylabel('MAE (a.u.)')
    plt.savefig(fig_save_dir + 'MAE.png'%Npar, bbox_inches="tight")
    plt.savefig(fig_save_dir + 'MAE.svg'%Npar, bbox_inches="tight")
    plt.show()
    
    # plt.scatter(Npar_list, R_est_all.mean(axis=0))
    sns.violinplot(data=R_est_all, color='skyblue')
    plt.plot([0, len(Npar_list)], [1.3, 1.3], 'r--', linewidth=2, zorder=0)
    plt.xticks(ticks=tick_idx, labels = xtick_lab)
    plt.xlabel('Num. of ensemble')
    # plt.yticks(ticks=np.arange(1, 8, 2))
    # plt.ylim(1.4, 2.2)
    plt.ylabel('estimated noise covariance')
    plt.savefig(fig_save_dir + 'mean_noise_cov.png'%Npar, bbox_inches="tight")
    plt.savefig(fig_save_dir + 'mean_noise_cov.svg'%Npar, bbox_inches="tight")
    plt.show()
#%%
if __name__ == '__main__':
    main()
    #%%
    
    