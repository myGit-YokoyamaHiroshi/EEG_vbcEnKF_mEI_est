
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 06:01:17 2021

@author: H.yokoyama
"""

from IPython import get_ipython
from copy import deepcopy
from scipy import signal as sig
import matplotlib.pylab as plt
import numpy as np

from PyEMD import EMD
import matplotlib.pylab as plt
#%%
EMD.splineKind    = 'cubic'
EMD.nbsym         = 2
EMD.reduceScale   = 1.
EMD.scaleFactor   = 100
EMD.maxIteration  = 100
EMD.stop1         = 0.05
EMD.stop2         = 0.5
EMD.stop3         = 0.05
EMD.DTYPE         = np.float64
EMD.MAX_ITERATION = 500
EMD.extrema_detection = "parabol"
"""
This script is class of preprocessing function.
The following input arguments are required:
    EEG : 3D matrix, cotaining raw EEG signals, with size [time x ch x trials]
    Reref : 2D matrix, containing rereference signal or common average signal of EEGs, with size [time x trials]
    chanlabels : vector variable containg the labels of each channel with char varialbes form
    fs: value of sampling frequency
    filter_band : vector of filter band frequency using band pass filter function for hum-canceling
    nofsift : number of sift that is requred using EMD function

Requirements
  Operation has been confirmed only under the following environment.
   - Python 3.8.3
   - conda 4.8.4
   - Spyder 4.1.4
   - numpy 1.18.5
   - scipy 1.5.0
   - matplotlib 3.2.2
   - PyEMD (https://github.com/laszukdawid/PyEMD)
   - IPython 7.16.1
   
example of usase
    >> prepro = Preprocessing(EEG_raw, Reref, chanlabels, fs, np.array([59, 61]), nofsifts)
    >> prepro.rereference()
    >> EEG_denoise = prepro.denoising()
"""
#%%
class Preprocessing:
    def __init__(self, EEG, t, Ref_sig, chanlabels, fs, nofsifts, detrend = 1):
        
        self.EEG         = EEG
        self.Ref_sig     = Ref_sig
        self.t           = t
        self.fs          = fs
        self.chanlabels  = chanlabels
        # self.filter_band = filter_band
        self.nofsifts    = nofsifts
        self.detrend     = detrend
        
    def rereference(self): # common average reference
        if self.Ref_sig.size == 0:
            tmp      = deepcopy(self.EEG)
            ref      = tmp.mean(axis=0)
            self.EEG = tmp-ref[np.newaxis,:,:]
        else: # rereference using reference signal
            tmp      = deepcopy(self.EEG)
            ref      = deepcopy(self.Ref_sig)
            self.EEG = tmp-ref[:,np.newaxis,:]
        
        return self.EEG

    def EMD_denosing_Flandrin(self, sig, t):
        detrend           = self.detrend
        nofsifts          = self.nofsifts
        # signal, t, nofsifts, detrend = 0
        signal            = sig.reshape(-1, order='f')
        
        """
        Python implementation of EMD-thresholding method, proposed by Flandrin et al. (2015)
        In this script, the parameter settings to estimate the threshold is applied, 
        assuming the "hurst exponent H = 0.5, and confidence interval = 95%", based on the reference paper .
        
        %signal       : Noisy signal
        %t            : time stamp of each samples
        % REFERECIES:
        % [1] Flandrin, P., Gonçalves, P., & Rilling, G. (2005). 
        %     EMD equivalent filter banks, from interpretation to applications. 
        %     In Hilbert-Huang transform and its applications (pp. 57-74).
        % [2] Flandrin, P., Goncalves, P., & Rilling, G. (2004, September). 
        %     Detrending and denoising with empirical mode decompositions. 
        %     In 2004 12th European Signal Processing Conference (pp. 1581-1584). IEEE.
        """
        #%% set parameters
        a_h  =  0.474#0.460#
        b_h  = -2.449#-1.919#
        Beta =  0.719
        H    =  0.5 # hurst exponent
        
        # a_h  =  0.495
        # b_h  = -1.833
        # Beta =  1.025
        # H    =  0.8 # hurst exponent
        
        rho  = 2.01 + 0.2*(H-0.5) + 0.12*(H-0.5)**2;
        #%% ##### Estimate IMFs for signal + noise mixture
        X          = deepcopy(signal )
        
        emd        = EMD()    
        emd.FIXE   = nofsifts
        emd.FIXE_H = nofsifts
        IMFs       = emd(X, t);
        
        Num        = IMFs.shape[0]
        
        # print("EMD: caclulated")
        #%% ###############################################################
        #### Estimate the noise energy and confidence interval
        Wh         = np.zeros(Num)
        Wh[0]      = np.median(abs(IMFs[0,:])**2)
        
        k          = np.arange(1, Num+1, 1)
        C          = Wh[0]/Beta
        for i in range(1, len(Wh)):
            Wh[i]  = C * rho**(-2*(1-H)*k[i])
            
        CI        =  2**(a_h * k + b_h) + np.log2(Wh)# confidence interval of noise
        threshold = CI 
        ###### Thresholding 
        h_hat     = np.zeros(IMFs.shape)
        for i in range(1, IMFs.shape[0]):
            c = deepcopy(IMFs[i, :])
            E = np.log2(np.median(abs(c)**2))
            if E >= threshold[i]:
                h_hat[i,:] = c
        # print("noise CI: caclulated")
        #%% detrend
        if detrend == 1:
            means      = np.zeros(Num)
            for i in range(1, Num+1):
                means[i-1] = np.mean(np.sum(IMFs[:i, :], axis=0))
                
            means      = means/means.sum()
            diff_means = np.concatenate( (np.array([0]), np.diff(np.sign(means))))
            idx        = np.where((diff_means != 0) & (abs(means) >= 0.05))[0]#np.where(means <= -0.05)[0]#
            
            if len(idx) != 0:
                idx = idx[idx!=0].min()
                h_hat[idx:,:] = 0
        # print("EMD denoising: caclulated")
        #%% ##### Make partial reconstructed sig
        sig_denoise = np.sum(h_hat, axis=0)
        
        return sig_denoise

    def denoising(self):
        
    #%% parameter setting for band stop/pass filter
        ############################################   
        
        # filt          = self.filter_band
        fs            = self.fs
        
        # if type(filt) == np.float64:
        #     ##### FIR low pass filter
        #     cutoff      = filt  # Desired cutoff frequency, Hz
        #     trans_width = 5         # Width of transition from pass band to stop band, Hz
        #     numtaps     = int(fs) # Size of the FIR filter.
        #     b           = sig.remez(numtaps, [0, cutoff, cutoff + trans_width, 0.5*fs], [1, 0], Hz=fs)
        #     a           = 1
        # else:
        #     ##### Bandpass filter
        #     band        = filt
        #     trans_width = .1    # Width of transition from pass band to stop band, Hz
        #     numtaps     = 6000        # Size of the FIR filter.
        #     # edges       = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, 0.5*fs]
        #     # b           = sig.remez(numtaps, edges, [0, 1, 0], Hz=fs)
            
        #     b           = sig.firwin(numtaps, cutoff = band, fs=fs, width = trans_width, window = "hanning", pass_zero = False)
        #     a           = 1

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%% denoising with EMD-based algorithm
        Nt, Nch, Ntri = self.EEG.shape
        
        EEG_denoise   = np.zeros(self.EEG.shape)
        for ch in range(Nch):
            #%%
            tmp     = self.EEG[:, ch,:]
            tmp_v   = tmp.reshape(-1, order='f')
            
            # # ## lowpass filter or band pass filter
            # tmp_v   = tmp_v - tmp_v[0]
            # tmp_v   = sig.filtfilt(b, a, tmp_v.T).T
                        
            # ## bandstop filter (hum cancel)
            # tmp_v   = tmp_v - tmp_v.mean()
            # tmp_v   = sig.filtfilt(b_hum, a_hum, tmp_v.T).T
            
            
            if self.nofsifts == 0:
                denoise = tmp_v.reshape(tmp.shape, order='f')
            elif self.nofsifts != 0:
                #%% EMD denosing
                t       = self.t
                denoise = Preprocessing.EMD_denosing_Flandrin(self, tmp_v, t)
                # denoise = denoise - denoise[0]
                denoise = denoise.reshape(tmp.shape, order='f')

            
            EEG_denoise[:,ch,:] = denoise
            print('channel : %s ___ preprocessed.'% self.chanlabels[ch])
        ######################################################################
        return EEG_denoise
    
    def calc_fft_STFT(self, n_slide, flimit):
        data     = self.EEG # EEG: [Nt x Nch x Ntri]
        fs       = self.fs
        n_sample = data.shape[0]
        
        
        dt    = 1.0/fs
        freqs = np.linspace(0, 1.0/dt, n_sample) # list of frequency
        
        Spct  = Preprocessing.stft(data, n_sample, n_slide, fs) # freqs x time x trial
        Pwr   = np.sum(abs(Spct)**2, axis=1)
        
        idx_f = np.where(freqs >= flimit[0] and freqs <= flimit[1])[0]
        freqs = freqs[idx_f]
        Spct  = Spct[idx_f,:]
        Pwr   = Pwr[idx_f,:]
        
        return Pwr, Spct, freqs
    
    def stft(data2d, n_sample, n_slide, fs):
        print('stft...')
        # data2d: (time, trial)
        windowFunc = np.hanning(n_sample)[:, np.newaxis]
        loopNum= int((data2d.shape[0]-n_sample)/n_slide+1)
        
        for i in range(loopNum):
            #str_plot = "Segment " + str(i+1) + ": Now calculating..."
            #print(str_plot)
            oneSgmt = data2d[i*n_slide:i*n_slide+n_sample,:] #res (time, trial)
            newData = oneSgmt*windowFunc
            X = np.fft.fft(newData, axis=0)
            if i==0:
                resultFFT = X
            else:
                resultFFT = np.dstack((resultFFT, X))
                
        resultFFT2 = np.swapaxes(resultFFT, 1,2)
        
        return resultFFT2