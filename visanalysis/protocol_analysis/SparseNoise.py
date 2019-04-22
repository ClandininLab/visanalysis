#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from visanalysis import protocol_analysis as pa

class SparseNoiseAnalysis(pa.ProtocolAnalysis.BaseAnalysis):
    def __init__(self):
        super().__init__() #call the parent class init method first

    def initializeAnalysis(self):
        max_phi=128
        max_theta=256
        output_shape = (max_phi, max_theta)
        
        self.num_phi = int(180 / self.ImagingData.epoch_parameters[0]['phi_period'])
        self.num_theta = int(360 / self.ImagingData.epoch_parameters[0]['theta_period'])

        # Get noise_frames at each imaging acquisition time point
        noise_frames = np.empty(shape=(self.num_phi, self.num_theta, self.ImagingData.roi_response[0].shape[1]), dtype=float)
        noise_frames[:] = np.nan
        for stack_ind, stack_time in enumerate(self.ImagingData.response_timing['stack_times']): #msec
            start_inds = np.where(stack_time>self.ImagingData.stimulus_timing['stimulus_start_times'])[0]
            stop_inds = np.where(stack_time<self.ImagingData.stimulus_timing['stimulus_end_times'])[0]
            if start_inds.size==0 or stop_inds.size==0: #before or after first/last epoch
                noise_frames[:,:,stack_ind] = self.ImagingData.run_parameters['idle_color']
            else:
                start_ind = start_inds[-1]
                stop_ind = stop_inds[0]
                if start_ind == stop_ind: #inside an epoch. Get noise grid
                    epoch_ind = int(start_ind)
                    start_seed = self.ImagingData.epoch_parameters[epoch_ind]['start_seed']
                    if 'rand_min' in self.ImagingData.epoch_parameters[epoch_ind]:
                        rand_min = self.ImagingData.epoch_parameters[epoch_ind]['rand_min']
                        rand_max = self.ImagingData.epoch_parameters[epoch_ind]['rand_max']
                        sparseness = self.ImagingData.epoch_parameters[epoch_ind]['sparseness']
                    else: #TODO: remove this eventually. For beta version of protocol code
                        rand_min = float(self.ImagingData.epoch_parameters[0]['distribution_data'].split('rand_min')[-1][3:6])
                        rand_max = float(self.ImagingData.epoch_parameters[0]['distribution_data'].split('rand_max')[-1][3:6])
                        sparseness = float(self.ImagingData.epoch_parameters[0]['distribution_data'].split('sparseness')[-1][3:7])
                        
                    update_rate = self.ImagingData.epoch_parameters[epoch_ind]['update_rate'] #Hz
                    current_time = (stack_time-self.ImagingData.stimulus_timing['stimulus_start_times'][epoch_ind]) / 1e3 #msec->sec
                    
                    seed = int(round(start_seed + current_time*update_rate))
                    np.random.seed(seed)
                    mean_p = sparseness
                    tail_p = (1.0-sparseness)/2
                    face_colors = np.random.choice([rand_min, (rand_min + rand_max)/2 , rand_max],
                                                        size=output_shape,
                                                        p = (tail_p, mean_p, tail_p))
                    #generates grid using max phi/theta, shader only takes the values that it needs to make the grid 
                    face_colors = face_colors[0:self.num_phi,0:self.num_theta]
                    
                    noise_frames[:,:,stack_ind] = face_colors
                    
                else: #between epochs, put up idle color
                    noise_frames[:,:,stack_ind] = self.ImagingData.run_parameters['idle_color']


        self.noise_frames = noise_frames

        pre_time = self.ImagingData.run_parameters['pre_time'] * 1e3 #msec
        tail_time = self.ImagingData.run_parameters['tail_time'] * 1e3 #msec
        stim_time = self.ImagingData.run_parameters['stim_time'] * 1e3 #msec
        raw_response = self.ImagingData.roi_response[0].copy()
        
        #do basic baseline first based on first pre_time
        pre_end = self.ImagingData.stimulus_timing['stimulus_start_times'][0]
        pre_start = pre_end - pre_time
        pre_inds = np.where(np.logical_and(self.ImagingData.response_timing['stack_times'] < pre_end,
                                   self.ImagingData.response_timing['stack_times'] >= pre_start))[0]
        baseline = np.mean(raw_response[:,pre_inds],axis = 1)
        self.current_response = (raw_response-baseline) / baseline
        
        #Recalcuate baseline for points within epoch based on each epochs pre-time
        #   Accounts for some amount of drift over long recordings (e.g. bleaching)
        for eInd,stimulus_start in enumerate(self.ImagingData.stimulus_timing['stimulus_start_times']):
            epoch_start = stimulus_start - pre_time
            epoch_end = epoch_start + pre_time + stim_time + tail_time
            pre_inds = np.where(np.logical_and(self.ImagingData.response_timing['stack_times'] < stimulus_start,
                                   self.ImagingData.response_timing['stack_times'] >= epoch_start))[0]
            baseline = np.mean(raw_response[:,pre_inds], axis = 1)
            
            
            epoch_inds = np.where(np.logical_and(self.ImagingData.response_timing['stack_times'] < epoch_end,
                                   self.ImagingData.response_timing['stack_times'] >= epoch_start))[0]
            self.current_response[:,epoch_inds] = (raw_response[:,epoch_inds] - baseline) / baseline

    def getStrfByFft(self, normalize_by_stimulus_autocorrelation=False):
        """
        Get STRF by FFT filter finder
        """
        filter_time = 3000 #ms
        filter_len = int(filter_time / self.ImagingData.response_timing['sample_period'])
        sample_rate = 1 / (self.ImagingData.response_timing['sample_period'] / 1e3)

        self.strf_fft = np.empty(shape=(self.num_phi, self.num_theta, filter_len), dtype=float)
        self.strf_fft[:] = np.nan
        for phi in range(self.num_phi):
            for theta in range(self.num_theta):
                # bring stim back to mean so extrapolation doesn't hang out  if last frame is a +/-
                current_stim = self.noise_frames[phi, theta, :]
                
                filter_fft = np.mean(np.fft.fft(self.current_response) * np.conj(np.fft.fft(current_stim)), axis = 0)
                if normalize_by_stimulus_autocorrelation:
                    norm_factor = np.fft.fft(current_stim) * np.conj(np.fft.fft(current_stim))
                    filter_fft = filter_fft / norm_factor
                    
                    freqcutoff = self.ImagingData.epoch_parameters[0]['update_rate'] #hz
                    frequency_cutoff = int(freqcutoff/(sample_rate/len(current_stim)))
                    filter_fft[frequency_cutoff:len(current_stim)-frequency_cutoff] = 0
                
                new_filt = np.real(np.fft.ifft(filter_fft))[0:filter_len]
                self.strf_fft[phi,theta,:] = new_filt
                self.fft_time = np.arange(0,filter_len) / sample_rate
  
    def getStrfByForwardCorrelation(self):
        """
        Get STRF by forward correlation with stimulus
        """
        self.delays = np.linspace(0,3,100) #sec
        
        self.strf_on = np.empty(shape=(self.num_phi, self.num_theta, len(self.delays)), dtype=float)
        self.strf_on[:] = 0   
        self.strf_off = self.strf_on.copy()
        self.strf_on_off = self.strf_on.copy()
        
        for phi in range(self.num_phi):
            for theta in range(self.num_theta):
                current_stim = self.noise_frames[phi, theta, :]
                on_inds = np.where(current_stim == 1.0)[0]
                off_inds = np.where(current_stim == 0.0)[0]
                
                on_times = self.ImagingData.response_timing['stack_times'][on_inds] / 1e3 #msec -> sec
                off_times = self.ImagingData.response_timing['stack_times'][off_inds] / 1e3 #msec -> sec
                #interpolate response
                f_resp = interpolate.interp1d(self.ImagingData.response_timing['stack_times']/1e3,
                                              self.current_response,
                                              kind = 'linear', 
                                              fill_value = 'extrapolate')
                
                on_prod = []
                off_prod = []
                on_off_prod = []
                
                for d_ind, delay in enumerate(self.delays):
                    on_resp = np.mean(f_resp(on_times+delay))
                    off_resp = np.mean(-f_resp(off_times+delay))
                    on_off_resp = np.mean(np.append(f_resp(on_times+delay), -f_resp(off_times+delay)))
                    
                    on_prod.append(on_resp)
                    off_prod.append(off_resp)
                    on_off_prod.append(on_off_resp)
         
                self.strf_on[phi,theta,:] = on_prod
                self.strf_off[phi,theta,:] = off_prod
                self.strf_on_off[phi,theta,:] = on_off_prod
                
    def plotStrfMultiPanel(self):
        fh = plt.figure(figsize=(10,4))
        target_delays = np.linspace(0,2,9)
        for tInd, delay_time in enumerate(target_delays):            
            new_ax = fh.add_subplot(4,9,tInd+1)
            add_strf(self.strf_off_z,self.delays,delay_time,new_ax)
            if tInd == 0: new_ax.set_ylabel('off')
            new_ax.set_title(str(int(delay_time*1e3)))
            
            new_ax = fh.add_subplot(4,9,9+tInd+1)
            add_strf(self.strf_on_z,self.delays,delay_time,new_ax)
            if tInd == 0: new_ax.set_ylabel('on')

            new_ax = fh.add_subplot(4,9,18+tInd+1)
            add_strf(self.strf_on_off_z,self.delays,delay_time,new_ax)
            if tInd == 0: new_ax.set_ylabel('on, off')
            
            new_ax = fh.add_subplot(4,9,27+tInd+1)
            add_strf(self.strf_fft_z,self.fft_time,delay_time,new_ax)
            if tInd == 0: new_ax.set_ylabel('FFT')
        fh.tight_layout()

    def plotStrf(self, strf, delay_time = 0.3, fig_handle = None):
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(3.5,2.75))
        new_ax = fig_handle.add_subplot(211)
        hmap = add_strf(strf,self.delays,delay_time,new_ax)

        new_ax.set_xticks([30, 45, 60, 75, 90])
        new_ax.set_yticks([150, 130, 110, 90])
        fig_handle.colorbar(hmap, ax=new_ax)
        fig_handle.tight_layout()

    def plotTemporalFilter(self,pullPhi = 110, pullTheta = 70, fig_handle = None):
        if fig_handle is None:
            fig_handle = plt.figure(figsize=(3.5,2.75))
        
        phi_step = self.ImagingData.epoch_parameters[0]['phi_period']
        theta_step = self.ImagingData.epoch_parameters[0]['theta_period']
        phis = np.arange(0,180,phi_step)
        thetas = np.arange(0,360,theta_step)
        phi_ind = np.argmin(np.abs(pullPhi - phis))
        theta_ind = np.argmin(np.abs(pullTheta - thetas))
                
        new_ax = fig_handle.add_subplot(212)
        off_filter = self.strf_off_z[phi_ind,theta_ind,:]
        on_filter = self.strf_on_z[phi_ind,theta_ind,:]
        on_off_filter = self.strf_on_off_z[phi_ind,theta_ind,:]
        fft_filter = self.strf_fft_z[phi_ind,theta_ind,:]

        new_ax.plot(self.delays,off_filter, 'r')
#        new_ax.plot(self.delays,on_filter, 'b')
#        new_ax.plot(self.delays,on_off_filter, 'k')
#        
#        ax2 = new_ax.twinx()
#        ax2.plot(self.fft_time,fft_filter)
#        ax2.set_ylabel('a.u.', color=self.plot_colors[0])
#        ax2.tick_params('y', colors=self.plot_colors[0])
        fig_handle.tight_layout()
        new_ax.set_xlabel('Time (s)')
        
    def makeStimulusFrameFigure(self):
        fig_handle = plt.figure(figsize=(10,3))
        extent=[0, 360,
            180, 0]
        for ff in range(10):
            ax = fig_handle.add_subplot(1,10,ff+1)
            ax.imshow(self.noise_frames[:,:,30 + 5 * ff], extent=extent, cmap=plt.cm.Greys,interpolation='none')
            ax.set_xlim([30, 90])
            ax.set_ylim([150, 90])
            ax.set_xticks([])
            ax.set_yticks([])
        
    def getZScoredStrfs(self):
        self.strf_off_z = z_score(self.strf_off)
        self.strf_on_z = z_score(self.strf_on)
        self.strf_on_off_z = z_score(self.strf_on_off)
        self.strf_fft_z = z_score(self.strf_fft)
 
        
    def doAnalysis(self, eg_trace_ind=0, export_to_igor_flag = False):
        self.getStrfByFft()
        self.getStrfByForwardCorrelation()
        self.getZScoredStrfs()
        self.plotAnalysisResults()
        
    def plotAnalysisResults(self, pullPhi = 110, pullTheta = 70, strf = None):
        if strf is None:
            strf = self.strf_off_z
        self.plotStrfMultiPanel()
        fig_handle = plt.figure(figsize=(3.5,2.75))
        self.plotStrf(strf, delay_time = 0.3, fig_handle = fig_handle)
        self.plotTemporalFilter(pullPhi = 120, pullTheta = 60, fig_handle = fig_handle)
        
    def doOnlineAnalysis(self, fig_handle = None):
        self.getStrfByForwardCorrelation()
        self.plotStrf(self.strf_on_off, delay_time = 0.3, fig_handle = fig_handle)
        fig_handle.canvas.draw()


def add_strf(strf,time_vector,delay_time,new_ax):
    extent=[0, 360,
            180, 0]

    pull_ind = np.argmin(np.abs(time_vector-delay_time))
    clim = np.max(np.abs(strf))
    hmap = new_ax.imshow(strf[:,:,pull_ind],vmin = np.min(strf), vmax = np.max(strf),
                  extent=extent, cmap=plt.cm.RdBu,interpolation='none')
    new_ax.set_xlim([30, 90])
    new_ax.set_ylim([150, 90])
    new_ax.set_xticks([])
    new_ax.set_yticks([])
    return hmap

def z_score(strf):
    strf_z = (strf - np.mean(strf)) / np.std(strf)
    
    return strf_z
