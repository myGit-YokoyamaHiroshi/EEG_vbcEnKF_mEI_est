# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:10:06 2023

@author: H.Yokoyama
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

from my_modules.preprocessing import Preprocessing
from tqdm import tqdm, trange
from scipy import signal as sig
from statsmodels.graphics.gofplots import qqplot 
from scipy.stats import norm, uniform 
from matplotlib.collections import LineCollection
import scipy.linalg
import math
import seaborn as sns

import numpy as np
import joblib
import random

import pyedflib
#%%
def plot_p_matrix(pmat, ax, mask=None):
    
    s = sns.heatmap(pmat, ax=ax,
                   mask=mask,
                   vmin=0, vmax=1, center=0,
                   cmap='copper', linewidths=1, linecolor='black', 
                   cbar_kws={'orientation': 'vertical', 'label': '$p$-value (Bonferroni-corrected)'})
    s.set_xticklabels(s.get_xticklabels(), rotation =45)
    
    N, M = pmat.shape
    
    for i in range(N):
        for j in range(i, N):
            if i != j:
                if (np.isnan(pmat.values[i,j])) or (pmat.values[i,j]>0.05):
                    plt.text(i+0.5, j+0.5,'$%.4f$\n(n.s.)'%pmat.values[i,j], va='center', ha='center')
                else:
                    plt.text(i+0.5, j+0.5,'$%.4f^{*}$'%pmat.values[i,j], va='center', ha='center')

def plot_effectsize_matrix(ef_mat, pmat, ax, mask=None):
    
    s = sns.heatmap(ef_mat, ax=ax,
                   mask=mask,
                   vmin=0, vmax=.3, #center=.2,
                   cmap='Reds', linewidths=1, linecolor='black', 
                   cbar_kws={'orientation': 'vertical', 'label': 'pairwise effect size (Cramer\'s $\phi^2$)'})
    s.set_xticklabels(s.get_xticklabels(), rotation =45)
    
    N, M = ef_mat.shape
    
    for i in range(N):
        for j in range(i, N):
            if i != j:
                plt.text(i+0.5, j+0.5,'%.4f'%ef_mat.values[i,j], va='center', ha='center')
    

############################################################################## 
#%% check the directory and get the file name automatically

fullpath   = './save_data/subject_list.npy'
load_dict  = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
list_sbj   = load_dict['list_subject']

del load_dict
#%%
fig_save_dir = current_path + '/figures/' 
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

data_save_dir = current_path + '/save_data/est_result/' 
if os.path.exists(data_save_dir)==False:  # Make the directory for data saving
    os.makedirs(data_save_dir)
#%%

EIave_state = np.zeros((len(list_sbj), 5))
idx_order  = np.array([1, -1, -2, -3, 0])

cnt = 0
for sbjID in list_sbj:
    #%% load data
    fname        = sbjID + 'EI_est_data.npy'  
    fullpath     = data_save_dir + fname 
    datadict     = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
    
    model        = datadict['model']
    eeg_obs      = datadict['eeg_observe']
    eeg_pred     = datadict['eeg_pred']    
    param_pred   = datadict['param_pred']
    state        = datadict['sleep_state']
    time         = datadict['time'] # unit: s
    fs           = datadict['fs'] 
    R            = datadict['R']
    
    REM_st       = datadict['idx_REM_st']
    #%%
    fig = plt.figure(figsize=(14, 7))
    gs  = fig.add_gridspec(2,3)
    plt.subplots_adjust(wspace=0.6, hspace=0.8)
    
    
    time_range = np.array([[-0.05,  0.05],
                           [(time[REM_st]/2)/60 - 0.05, (time[REM_st]/2)/60 + 0.05],
                           [time[REM_st]/60 - 0.05, time[REM_st]/60 + 0.05]
                           ])
    
    ax1 = fig.add_subplot(gs[0, 0:3])
    
    ax1.plot(time/60, eeg_obs, label='exact', zorder=1, alpha=0.7);
    ax1.plot(time[1:]/60, eeg_pred[:-1], label='predicted', zorder=2, alpha=0.7);
    ax1.set_xlabel('time (min)')
    ax1.set_ylabel('amplitude ($\\mu V$)')
    ax1.set_xlim(-20, time[-1]/60)
    ax1.set_ylim(-220, 220)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
    ax1.axvspan(time[REM_st]/(60), time[-1]/(60), color='gray', alpha=0.3, zorder=3)
    ax1.plot([0,0], [-300, 300], 'r', linestyle='--', linewidth=4, zorder=4)
    
    ax1.plot(time_range.T, [-218,-218], linewidth=6, zorder=5, c='k')
    
    ############
    
    for i in range(3):
        axn = fig.add_subplot(gs[1, i])
        axn.plot(time/60, eeg_obs, label='exact', zorder=1, linewidth=1.5, alpha=0.7);
        axn.plot(time[1:]/60, eeg_pred[:-1], label='predicted', zorder=2, alpha=0.7, linewidth=1.5);
        axn.set_xlabel('time (min)')
        
        axn.plot([0,0], [-300, 300], 'r', linestyle='--', linewidth=4, zorder=4)
        axn.axvspan(time[REM_st]/(60), time[-1]/(60), color='gray', alpha=0.3, zorder=3)
        
        axn.set_xlim(time_range[i,:])
        axn.set_ylim(-90, 90)
        # axn.set_xticks(np.arange(time_range[i,0], time_range[i,1]+0.09, 0.1))
        if i == 0:
            axn.set_ylabel('amplitude ($\\mu V$)')
    #######
    figname = fig_save_dir + sbjID + '_predicted_eeg'
    plt.savefig(figname + '.png', bbox_inches="tight")
    plt.savefig(figname + '.svg', bbox_inches="tight")
    plt.show()
    
    #%%
    win          = 0.5# unit s
    EIR          = param_pred[:,0]/(param_pred[:,0] + param_pred[:,2]) # = A/B
    
    # EIRsm = moving_average(EIR, int(fs*10))
    
    fig = plt.figure(figsize=(8, 10))
    gs  = fig.add_gridspec(3, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.1)
    
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(time/60, EIR, zorder=1, label='EI ratio')
    # ax1.plot(time[int(fs*10)-1:]/60, EIRsm, zorder=2, label='EI ratio (smooth)')
    # ax1.errorbar(t_epc, EIR_epc, yerr=SD_epc, 
    #              marker='o', mfc='black', ms=5, linestyle='-', 
    #              zorder=2)
    ax1.plot([0,0], [-300, 300], 'r', linestyle='--', linewidth=4, zorder=4)
    ax1.axvspan(time[REM_st]/(60), time[-1]/(60), color='gray', alpha=0.3, zorder=3)
    ax1.set_xlim(-20, time[-1]/60)
    ax1.set_ylim(-0.05, 1.0)
    # plt.xlim(-21, time[-1]/60)
    ax1.set_xticklabels([])
    ax1.set_ylabel('mE/I ratio (a.u.)')
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.grid()
    #######################
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(time/60, state, 'k', linestyle='-', linewidth=2)
    ax2.plot([0,0], [-4, 2], 'r', linestyle='--', linewidth=4, zorder=0)
    # ax2.axvspan(time[0]/(60), time[int(fs*baseline)]/(60), color='gray', alpha=0.3)
    ax2.axvspan(time[REM_st]/(60), time[-1]/(60), color='gray', alpha=0.3)
    ax2.set_yticks([-3, -2, -1, 0, 1], )
    ax2.set_yticklabels(['Stage 3/4', 'Stage 2', 'Stage 1', 'REM', 'Awake'])
    ax2.set_xlabel('time (min)')
    ax2.set_xlim(-20, time[-1]/60)#set_xlim(-3, time[-1]/60)
    ax2.set_ylim(-3.5, 1.5)
    plt.grid()
    ########
    figname = fig_save_dir + sbjID + '_EI_ratio'
    plt.savefig(figname + '.png', bbox_inches="tight")
    plt.savefig(figname + '.svg', bbox_inches="tight")
    
    plt.show()
    
    for st in idx_order:
        idx = np.where(idx_order==st)[0]
        EIave_state[cnt, idx] = np.nanmean(EIR[(state==st)])
    print(sbjID)    
    cnt += 1
#%%
fig, ax = plt.subplots(figsize=(8,8))
qqplot(EIave_state.reshape(-1)[np.isnan(EIave_state.reshape(-1))==False], 
       dist=norm, fit=True,line="r", ax=ax) 
ax.axis('equal')
figname = fig_save_dir + 'qq_plot'
plt.savefig(figname + '.png', bbox_inches="tight")
plt.savefig(figname + '.svg', bbox_inches="tight")
plt.show()


#%%
import pandas as pd
from scipy import stats
from my_modules.dunn import  my_posthoc_dunn

state_idx = np.tile(np.array(['Awake', 'Stage 1', 'Stage 2', 'Stage 3/4', 'REM']), (len(list_sbj),1))
state_idx = state_idx.reshape(-1)

subject   = np.tile(list_sbj, (5, 1)).T
subject   = subject.reshape(-1)
#%% Kruskal-Wallis one-way analysis (nonparametric one-way anova) and multiple comparison
EIave_vec          = EIave_state.reshape(-1)
table_kruskal      = stats.kruskal(EIave_vec[state_idx=='Awake'], EIave_vec[state_idx=='Stage 1'], EIave_vec[state_idx=='Stage 2'], EIave_vec[state_idx=='Stage 3/4'], EIave_vec[state_idx=='REM'], nan_policy='omit')
df_kruskal         = pd.DataFrame(table_kruskal)
df_kruskal.index   = ['statistic', 'pvalue']
df_kruskal.columns = ['Kruskal-Wallis test']


state_list = ['Awake', 'Stage 1', 'Stage 2', 'Stage 3/4', 'REM']
data = [list(EIave_vec[state_idx==state_list[i]],) for i in range(len(state_list))]

df_dnn, df_dnn_stat, df_dnn_effect = my_posthoc_dunn(data, p_adjust = 'bonferroni')
df_dnn.index   = state_list
df_dnn.columns = state_list

df_dnn_stat.index   = state_list
df_dnn_stat.columns = state_list

df_dnn_effect.index   = state_list
df_dnn_effect.columns = state_list

# table_dnn      = sp.posthoc_dunn(data, p_adjust = 'bonferroni')
# df_dnn         = pd.DataFrame(table_dnn)
# df_dnn.index   = state_list
# df_dnn.columns = state_list

n              = len(list_sbj)*len(state_list)
k              = len(state_list)
eta_sq         = (table_kruskal[0] - k + 1)/(n - k)

df_kruskal.loc['effect size'] =  eta_sq

df_kruskal.to_csv('kruskal_result.csv')
df_dnn.to_csv('dnn_result.csv')
df_dnn_effect.to_csv('dnn_result_effectsize.csv')
df_dnn_stat.to_csv('dnn_result_stats.csv')

print('--------------')
print(df_kruskal)
print('')
print(df_dnn)
#%%
fig = plt.figure(figsize=(40,9))
gs  = fig.add_gridspec(1,3)
plt.subplots_adjust(wspace=0.4, hspace=0.8)
    
    
ax1 = fig.add_subplot(gs[0, 0])
# sns.violinplot(data=EIave_state, color='skyblue', inner=None)
plt.violinplot([val[np.isnan(val)==False]  for val in EIave_state.T], 
               positions=np.arange(0,len(state_list)), 
               showextrema=True, showmedians=True)
sns.stripplot(data=EIave_state, jitter=True, color='blue')
ax1.set_xticks(ticks=np.arange(5))
ax1.set_xticklabels(state_list, rotation=45)
ax1.set_ylabel('mean of mE/I ratio')
ax1.set_ylim(-0.1, 1.0)

ax2 = fig.add_subplot(gs[0, 1])
plot_p_matrix(df_dnn, ax2, mask=np.triu(df_dnn));

ax3 = fig.add_subplot(gs[0, 2])
plot_effectsize_matrix(df_dnn_effect, df_dnn, ax3, mask=np.triu(df_dnn_effect));


figname = fig_save_dir + 'stats_result'
plt.savefig(figname + '.png', bbox_inches="tight")
plt.savefig(figname + '.svg', bbox_inches="tight")
plt.show()
