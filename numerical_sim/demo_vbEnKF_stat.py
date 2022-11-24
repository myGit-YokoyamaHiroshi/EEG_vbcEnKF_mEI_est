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
        
        time_range = np.array([[ 5,  6],
                               [15, 16],
                               [25, 26]])
        
        ax1 = fig.add_subplot(gs[0, 0:3])
        ax1.plot(time, eeg, c='#1f77b4', label='exact', zorder=1, alpha=0.7, linewidth=1.5);
        ax1.plot(time[1:], eeg_pred[0,:-1], color='#ff7f0e', label='predicted', zorder=2, alpha=0.7, linewidth=1.5);
        [ax1.plot(time_range[i,:], [-0.8,-0.8], linewidth=3, c='k') for i in range(3)]
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('amplitude (a.u.)')
        ax1.set_ylim(-1, 18)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
        ############
        
        for i in range(3):
            axn = fig.add_subplot(gs[1, i])
            axn.plot(time, eeg, c='#1f77b4', label='exact', zorder=1, alpha=0.7, linewidth=1.5);
            axn.plot(time[:-1], eeg_pred[0,1:], color='#ff7f0e', label='estimated', zorder=2, alpha=0.7, linewidth=1.5);
            axn.set_xlabel('time (s)')
            axn.set_xlim(time_range[i,:])
            axn.set_ylim(-1, 18)
            if i == 0:
                axn.set_ylabel('amplitude (a.u.)')
        #######
        plt.savefig(fig_save_dir + 'eeg_Npar%d.png'%Npar, bbox_inches="tight")
        plt.savefig(fig_save_dir + 'eeg_Npar%d.svg'%Npar, bbox_inches="tight")
        plt.show()
        
        #%%
    
        
        fig = plt.figure(figsize=(10, 18))
        gs  = fig.add_gridspec(6,4,height_ratios=[1.5,0.01,1,1,1, 0.1])
        plt.subplots_adjust(wspace=0.8, hspace=0.8)
        ### mEI ratio
        mEI_true = param_true[:,0]/(param_true[:,0] + param_true[:,2])
        mEI_pred = param_pred[:,:,0]/(param_pred[:,:,0] + param_pred[:,:,2])
        
        ax = fig.add_subplot(gs[0, 0:])
        ax.plot(time, mEI_true, color='#1f77b4', label='exact', zorder=2, alpha=0.7, linewidth=2.5);
        lines = [np.column_stack([time, mEI_pred[j,:]]) for j in range(Ntri)]
        lc = LineCollection(lines, cmap='autumn', linewidth=1.5)
        lc.set_array(np.arange(0, Ntri))
        lc.set_alpha(0.5)
        line = ax.add_collection(lc) #add to the subplot
        # ax.plot(time, mEI_pred.T, color='#ff7f0e', label='estimated', zorder=1, alpha=0.7, linewidth=1.5);
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude (a.u.)')
        ax.set_ylim(0.05, 0.4)
        ax.legend(['exact'],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
        ax.set_title('mEI ratio\n$mEI = A(t)/(A(t)+B(t))$')
        
        # add color bar below main figure panel
        ax_cb = fig.add_subplot(gs[1, 1:3])
        divider = make_axes_locatable(ax_cb)
        cax = divider.new_vertical(size='10%', pad=0.5, pack_start = True)
        fig.add_axes(cax)
        fig.colorbar(line, cax = cax, label='Num. of trial',  orientation='horizontal')
        ax_cb.tick_params(axis='x', colors='white') #setting up X-axis tick color to white
        ax_cb.tick_params(axis='y', colors='white') #setting up Y-axis tick color to white
        ax_cb.spines['bottom'].set_color('white')
        ax_cb.spines['top'].set_color('white')
        ax_cb.spines['left'].set_color('white')
        ax_cb.spines['right'].set_color('white')
        
        ### parameter A
        ax = fig.add_subplot(gs[2, 0:2])
        ax.plot(time, param_true[:,0], c='#1f77b4', label='exact', zorder=2, alpha=0.7, linewidth=2.5);
        lines = [np.column_stack([time, param_pred[j,:,0]]) for j in range(Ntri)]
        lc = LineCollection(lines, cmap='autumn', linewidth=1.5)
        lc.set_array(np.arange(0, Ntri))
        lc.set_alpha(0.5)
        line = ax.add_collection(lc) #add to the subplot
        # ax.plot(time, param_pred[:,:,0].T, c='#ff7f0e', label='estimated', zorder=1, alpha=0.7, linewidth=1.5);
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude (a.u.)')
        ax.set_ylim(2, 7)
        ax.set_title('$A(t)$')
        
        ### parameter B
        ax = fig.add_subplot(gs[2, 2:])
        ax.plot(time, param_true[:,2], c='#1f77b4', label='exact', zorder=2, alpha=0.7, linewidth=2.5);
        lines = [np.column_stack([time, param_pred[j,:,2]]) for j in range(Ntri)]
        lc = LineCollection(lines, cmap='autumn', linewidth=1.5)
        lc.set_array(np.arange(0, Ntri))
        lc.set_alpha(0.5)
        line = ax.add_collection(lc) #add to the subplot
        # ax.plot(time, param_pred[:,:,2].T, c='#ff7f0e', label='estimated', zorder=1, alpha=0.7, linewidth=1.5);
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude (a.u.)')
        ax.set_ylim(10, 30)
        ax.set_title('$B(t)$')
        
        
        ### parameter a
        ax = fig.add_subplot(gs[3, 0:2])
        ax.plot(time, param_true[:,1], c='#1f77b4', label='exact', zorder=2, alpha=0.7, linewidth=2.5);
        lines = [np.column_stack([time, param_pred[j,:,1]]) for j in range(Ntri)]
        lc = LineCollection(lines, cmap='autumn', linewidth=1.5)
        lc.set_array(np.arange(0, Ntri))
        lc.set_alpha(0.5)
        line = ax.add_collection(lc) #add to the subplot
        # ax.plot(time, param_pred[:,:,1].T, c='#ff7f0e', label='estimated', zorder=1, alpha=0.7, linewidth=1.5);
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude (a.u.)')
        ax.set_ylim(80, 120)
        ax.set_title('$a(t)$')
        
        ### parameter b
        ax = fig.add_subplot(gs[3, 2:])
        ax.plot(time, param_true[:,3], c='#1f77b4', label='exact', zorder=2, alpha=0.7, linewidth=2.5);
        lines = [np.column_stack([time, param_pred[j,:,3]]) for j in range(Ntri)]
        lc = LineCollection(lines, cmap='autumn', linewidth=1.5)
        lc.set_array(np.arange(0, Ntri))
        lc.set_alpha(0.5)
        line = ax.add_collection(lc) #add to the subplot
        # ax.plot(time, param_pred[:,:,3].T, c='#ff7f0e', label='estimated', zorder=1, alpha=0.7, linewidth=1.5);
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude (a.u.)')
        ax.set_ylim(40, 70)
        ax.set_title('$b(t)$')
        
        ### parameter p
        ax = fig.add_subplot(gs[4, 1:3])
        ax.plot(time, param_true[:,4], c='#1f77b4', zorder=1, alpha=0.7, linewidth=1.5);
        lines = [np.column_stack([time, param_pred[j,:,4]]) for j in range(Ntri)]
        lc = LineCollection(lines, cmap='autumn', linewidth=1.5)
        lc.set_array(np.arange(0, Ntri))
        lc.set_alpha(0.5)
        line = ax.add_collection(lc) #add to the subplot
        # ax.plot(time, param_pred[:,:,4].T, c='#ff7f0e', zorder=2, alpha=0.7, linewidth=1.5);
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude (a.u.)')
        ax.set_ylim(60, 420)
        ax.set_title('$p(t)$')
        ax.legend(['exact'],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
        
        # add color bar below main figure panel
        ax_cb = fig.add_subplot(gs[5, 1:3])
        divider = make_axes_locatable(ax_cb)
        cax = divider.new_vertical(size='10%', pad=0.5, pack_start = True)
        fig.add_axes(cax)
        fig.colorbar(line, cax = cax, label='Num. of trial',  orientation='horizontal')
        ax_cb.tick_params(axis='x', colors='white') #setting up X-axis tick color to white
        ax_cb.tick_params(axis='y', colors='white') #setting up Y-axis tick color to white
        ax_cb.spines['bottom'].set_color('white')
        ax_cb.spines['top'].set_color('white')
        ax_cb.spines['left'].set_color('white')
        ax_cb.spines['right'].set_color('white')
        #######
        plt.savefig(fig_save_dir + 'EIparam_Npar%d.png'%Npar, bbox_inches="tight")
        plt.savefig(fig_save_dir + 'EIparam_Npar%d.svg'%Npar, bbox_inches="tight")
        plt.show()
        #%%
        cnt += 1
    #%%
    fig_save_dir = current_path + '/figures/'
    
    tick_idx = np.where(np.mod(Npar_list, 100)==0)[0]
    
    xtick_lab = []
    for idx in tick_idx:
        xtick_lab.append(str(Npar_list[idx]))
    
    plt.violinplot(MAE, positions=np.arange(0,len(Npar_list)), showextrema=True, showmedians=True)
    plt.xticks(ticks=tick_idx, labels = xtick_lab)
    plt.xlabel('Num. of ensemble')
    plt.ylabel('MAE (a.u.)')
    plt.savefig(fig_save_dir + 'MAE.png'%Npar, bbox_inches="tight")
    plt.savefig(fig_save_dir + 'MAE.svg'%Npar, bbox_inches="tight")
    plt.show()
    
    plt.violinplot(R_est_all, positions=np.arange(0,len(Npar_list)), showextrema=True, showmedians=True)
    plt.plot([0, len(Npar_list)], [1.3, 1.3], 'r--', linewidth=2, zorder=0, alpha=0.7)
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
    
    