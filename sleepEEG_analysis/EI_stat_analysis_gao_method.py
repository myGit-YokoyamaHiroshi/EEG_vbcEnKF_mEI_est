#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:52:33 2022

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

from my_modules.preprocessing import Preprocessing
from tqdm import tqdm, trange
from scipy import signal as sig
from statsmodels.graphics.gofplots import qqplot 
from scipy.stats import norm, uniform 
import scipy.linalg
import math
import seaborn as sns

import numpy as np
import joblib
import random

import pyedflib
#%%
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def epoch_average(time, value, win):
    td       = np.mean(time[1:]-time[:-1])
    t_epc    = np.arange(time[0], time[-1], win)
    interval = np.concatenate((t_epc[:-1, np.newaxis], t_epc[1:, np.newaxis]), axis=1)
    Nepc     = interval.shape[0]
    
    ave_epc  = np.zeros(Nepc)
    sd_epc   = np.zeros(Nepc)
    for epc in range(Nepc):
        idx          = np.where((time>interval[epc,0]) & (time<=interval[epc,1]))[0]
        ave_epc[epc] = np.nanmean(value[idx])
        sd_epc[epc]  = np.nanstd(value[idx])
    
    t_epc = t_epc[1:]
    return t_epc, ave_epc, sd_epc

def my_progress_bar(t, Nt):
    bar_template = "\r{0}%[{1}] {2}/{3}"
    bar = "#" * round((t/Nt)*100) + " " * (100 - round((t/Nt)*100))
    print(bar_template.format(round((t/Nt)*100), bar, t, Nt), end="")
    
    
def plot_cor_matrix(corr, ax, mask=None):
    # f, ax = plt.subplots(figsize=(12, 9))
    
    # corr[corr>0.05] = 1
    
    s = sns.heatmap(corr, ax=ax,
                   mask=mask,
                   # cosmetics
                   vmin=0, vmax=1, center=0,
                   cmap='copper', linewidths=1, linecolor='black', 
                   cbar_kws={'orientation': 'vertical', 'label': '$p$-value (Bonferroni-corrected)'})
    s.set_xticklabels(s.get_xticklabels(), rotation =45)
    
    N, M = df_dnn.shape
    
    for i in range(N):
        for j in range(i, N):
            if i != j:
                if (np.isnan(corr.values[i,j])) or (corr.values[i,j]>0.05):
                    plt.text(i+0.5, j+0.5,'n.s.', va='center', ha='center')
                else:
                    plt.text(i+0.5, j+0.5,'%.4f'%corr.values[i,j], va='center', ha='center')
    
    # return f, ax

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

data_save_dir = current_path + '/save_data/gao_method/' 
if os.path.exists(data_save_dir)==False:  # Make the directory for data saving
    os.makedirs(data_save_dir)
#%%

EIave_state = np.zeros((len(list_sbj), 5))
idx_order  = np.array([1, -1, -2, -3, 0])

cnt = 0
for sbjID in list_sbj:
    #%% load data
    fname        = sbjID + 'EI_est_gao_method.npy'  
    fullpath     = data_save_dir + fname 
    datadict     = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
    
    eeg_obs      = datadict['eeg_observe']  
    state        = datadict['sleep_state']
    time         = datadict['time'] # unit: s
    fs           = datadict['fs'] 
    
    REM_st       = datadict['idx_REM_st']
    EIslope      = datadict['EIslope']
    coeff        = datadict['coeff'] 
    
    del datadict
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
    ax1.set_ylabel('mean E/I slope (a.u.)')
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
    figname = fig_save_dir + sbjID + '_EIslope_gao_method'
    plt.savefig(figname + '.png', bbox_inches="tight")
    plt.savefig(figname + '.svg', bbox_inches="tight")
    
    plt.show()
    #%%
    for st in idx_order:
        idx = np.where(idx_order==st)[0]
        EIave_state[cnt, idx] = np.nanmean(EIslope[(state==st)])
    print(sbjID)    
    cnt += 1
#%%
fig, ax = plt.subplots(figsize=(8,8))
qqplot(EIave_state.reshape(-1)[np.isnan(EIave_state.reshape(-1))==False], 
       dist=norm, fit=True,line="r", ax=ax) 
ax.axis('equal')
figname = fig_save_dir + 'qq_plot_gao'
plt.savefig(figname + '.png', bbox_inches="tight")
plt.savefig(figname + '.svg', bbox_inches="tight")
plt.show()


#%%
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

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
table_dnn      = sp.posthoc_dunn(data, p_adjust = 'bonferroni')
df_dnn         = pd.DataFrame(table_dnn)
df_dnn.index   = state_list
df_dnn.columns = state_list

n              = len(list_sbj)*len(state_list)
k              = len(state_list)
eta_sq         = (table_kruskal[0] - k + 1)/(n - k)

df_kruskal.loc['effect size'] =  eta_sq

df_kruskal.to_csv('kruskal_result_gao.csv')
df_dnn.to_csv('dnn_result_gao.csv')

print('--------------')
print(df_kruskal)
print('')
print(df_dnn)
#%%
fig = plt.figure(figsize=(26,9))
gs  = fig.add_gridspec(1,2)
plt.subplots_adjust(wspace=0.4, hspace=0.8)
    
    
ax1 = fig.add_subplot(gs[0, 0])
sns.violinplot(data=EIave_state, color='skyblue')
sns.stripplot(data=EIave_state, jitter=True, color='blue')
ax1.set_xticks(ticks=np.arange(5))
ax1.set_xticklabels(state_list, rotation=45)
ax1.set_ylabel('mean of E/I slope')
# ax1.set_ylim(-0.1, 1.0)

ax2 = fig.add_subplot(gs[0, 1])
plot_cor_matrix(df_dnn, ax2, mask=np.triu(df_dnn));
figname = fig_save_dir + 'stats_result_gao'
plt.savefig(figname + '.png', bbox_inches="tight")
plt.savefig(figname + '.svg', bbox_inches="tight")
plt.show()
