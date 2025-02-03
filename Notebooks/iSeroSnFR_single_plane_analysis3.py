# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:31:46 2023

@author: LS_User
"""


from scipy.ndimage import gaussian_filter1d
import pandas as pd
import glob
import tifffile
import h5py
import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 # important for vector text output
matplotlib.rcParams['ps.fonttype'] = 42  # important for vector text output
import matplotlib.pyplot as plt



from matplotlib.widgets import LassoSelector
from matplotlib import path
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.optimize import minimize

from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr,ttest_ind, ks_2samp,ttest_rel,ttest_1samp,spearmanr

plt.close('all')
root_dir=r'C:\Users\takashi.WISMAIN\Dropbox (Weizmann Institute)\DRN_gating\serotonin_imaging_raw'

dir_list=['05072023\single1_iSeroSnFR','05072023\single2_iSeroSnFR',
          '05092023\single1_iSeroSnFR','05092023\single2_iSeroSnFR',
          '05092023\single3_iSeroSnFR','05092023\single4_iSeroSnFR',
          '05212023\single1_iSeroSnFR','05212023\single2_iSeroSnFR',
          '05212023\single3_iSeroSnFR','05212023\single4_iSeroSnFR',
          '05212023\single5_iSeroSnFR','05212023\single6_iSeroSnFR',
          '05212023\single7_iSeroSnFR','05212023\single8_iSeroSnFR']




#%% normalize data

# for dd in range(len(dir_list)):
    
#     exp_dir=root_dir+r'\\'+dir_list[dd]
#     print(exp_dir)

#     def baseline_correction(timeseries):
        
#         background=100
#         rolling_window=2400
        
#         timeseries -= background
                       
#         baseline=pd.Series(timeseries).rolling(window=rolling_window,min_periods=int(rolling_window/2),center=True).quantile(0.3)                    
#         timeseries=timeseries/baseline    
#         timeseries_new=gaussian_filter1d(timeseries,3)-1
        
#         return timeseries_new
    
    
#     timeseries1 = np.load(exp_dir+r'\\hindbrainR_ROI_timeseries.npy')[()].astype(np.double)
#     for i in range(timeseries1.shape[1]):
#         timeseries1[:,i]=baseline_correction(timeseries1[:,i])
#         if i % 1000 == 0:
#             print(i)
#     np.save(exp_dir+r'\\hindbrainR_ROI_timeseries_norm.npy',timeseries1)
    
    
#     timeseries2 = np.load(exp_dir+r'\\hindbrainL_ROI_timeseries.npy')[()].astype(np.double)
#     for i in range(timeseries2.shape[1]):
#         timeseries2[:,i]=baseline_correction(timeseries2[:,i])
#         if i % 1000 == 0:
#             print(i)
#     np.save(exp_dir+r'\\hindbrainL_ROI_timeseries_norm.npy',timeseries2)
    
    
#     timeseries3 = np.load(exp_dir+r'\\tectumR_ROI_timeseries.npy')[()].astype(np.double)
#     for i in range(timeseries3.shape[1]):
#         timeseries3[:,i]=baseline_correction(timeseries3[:,i])
#         if i % 1000 == 0:
#             print(i)
#     np.save(exp_dir+r'\\tectumR_ROI_timeseries_norm.npy',timeseries3)
    
    
#     timeseries4 = np.load(exp_dir+r'\\tectumL_ROI_timeseries.npy')[()].astype(np.double)
#     for i in range(timeseries4.shape[1]):
#         timeseries4[:,i]=baseline_correction(timeseries4[:,i])
#         if i % 1000 == 0:
#             print(i)
#     np.save(exp_dir+r'\\tectumL_ROI_timeseries_norm.npy',timeseries4)

#%%

summary_activity = np.zeros((600,2,4,len(dir_list),2))
summary_activity_fast = np.zeros((90,2,4,len(dir_list),2))
summary_swim = np.zeros((2,200,len(dir_list),2))
prediction_info=np.zeros((len(dir_list),4,2))
fitting_info=np.zeros((len(dir_list),8))
convolve_kernel=np.zeros((11,))
convolve_kernel[6:] = 1/5

swim_statistics=[]

plt.figure(1,figsize=(18,6))
plt.figure(2,figsize=(18,6))
plt.figure(3,figsize=(18,6))
plt.figure(4,figsize=(18,4))

name_list=['hindbrainR','hindbrainL','tectumR','tectumL']


def baseline_correction(timeseries,filter_size=3):

    background=100
    rolling_window=2400

    timeseries -= background
           
    baseline=pd.Series(timeseries).rolling(window=rolling_window,min_periods=int(rolling_window/2),center=True).quantile(0.3)                    
    timeseries=timeseries/baseline    
    timeseries_new=gaussian_filter1d(timeseries,filter_size)-1

    return timeseries_new


for dd in range(len(dir_list)):
#for dd in range(12,13):
    
    exp_dir=root_dir+r'\\'+dir_list[dd]
    print(exp_dir)
    
    timeseries=[]
    #timeseries_norm=[]
    timeseries_inds_array=[]
    for i in range(len(name_list)):
        #timeseries.append(np.load(exp_dir+r'\\'+name_list[i]+'_ROI_timeseries.npy')[()].astype(np.double))
        #timeseries_norm.append(np.load(exp_dir+r'\\'+name_list[i]+'_ROI_timeseries_norm.npy')[()])
        timeseries_inds_array.append(np.load(exp_dir+r'\\'+name_list[i]+'_ROI_inds_array.npy')[()])
    
    swimdata = np.load(exp_dir+r'\\swimdata.npy',allow_pickle=True)[()]
    rawdata  = np.load(exp_dir+r'\\rawdata.npy',allow_pickle=True)[()]
    
    
    frames=np.where(np.diff((rawdata['ch3']>3).astype(np.int32))==1)[0]
    

    
    #%% Trial average
    
    period1_start=np.where((rawdata['envParam1'][:-1]==2) & (rawdata['envParam1'][1:]==1))[0]
    period2_start=np.where((rawdata['envParam1'][:-1]==1) & (rawdata['envParam1'][1:]==2))[0]
    
    period1_start=period1_start[period1_start>frames[1800]]
    period2_start=period2_start[period2_start>frames[1800]]
    
    trials=min(len(period1_start),len(period2_start))
    
    swim_startI=swimdata['swimStartIndT']
    swim_endI=swimdata['swimEndIndT']
    
    
    
    
    #%% calculate swimpower
    
    swim_powers=np.zeros((len(swim_startI),))
    back_velocity=np.zeros((len(swim_startI),2))
    gain=np.zeros((len(swim_startI),))
    swim_intervals=np.zeros((len(swim_startI),))
    
    for s in range(len(swim_startI)):
        
        fltCh1=swimdata['fltCh1'][swim_startI[s]:int(swim_endI[s]+1)]
        fltCh2=swimdata['fltCh2'][swim_startI[s]:int(swim_endI[s]+1)]
        backCh1=swimdata['back1'][swim_startI[s]:int(swim_endI[s]+1)]
        backCh2=swimdata['back2'][swim_startI[s]:int(swim_endI[s]+1)]
            
        swim_powers[s]   = max(0,fltCh1.sum()-backCh1.sum())+max(0,fltCh2.sum()-backCh2.sum())
        if s<(len(swim_startI)-1):
            tmp=rawdata['velocity'][int(swim_startI[s]-35*6):int(swim_startI[s+1]-35*6)].copy()
            tmp[tmp>0]=0
            back_velocity[s,0] = -tmp[:(swim_endI[s]-swim_startI[s])].sum()/6000
            back_velocity[s,1] = -tmp[(swim_endI[s]-swim_startI[s]):].sum()/6000
            # back_velocity[s,0] = -tmp[:100*6].sum()/6000
            # back_velocity[s,1] = -tmp[(swim_endI[s]-swim_startI[s]):(swim_endI[s]-swim_startI[s])+100*6].sum()/6000
        gain[s] = rawdata['envParam1'][swim_startI[s]]
        
        if s<(len(swim_startI)-1):
            swim_intervals[s]=swim_startI[s+1]-swim_endI[s]
            
        
        np.zeros((len(swim_startI),))
        
    # plt.figure()
    # plt.plot(back_velocity[:,0])
    # plt.plot(gain)
    # plt.plot(np.convolve(gain,convolve_kernel,'same'))
    
    # bbb
    

    
    
    
    #%%
    
    
    swim_trials = np.zeros((2,trials))
    swim_trace  = np.zeros((2,200,trials))
    lowgain_swim=[]
    highgain_swim=[]
    
    for i in range(trials):
        
        swim_inds1=np.where((swim_startI>period1_start[i]) & (swim_startI<(period1_start[i]+120000)))[0]
        
        for s in range(len(swim_inds1)):
            
            swim_trials[0,i] += swim_powers[swim_inds1[s]]
            lowgain_swim.append(swim_powers[swim_inds1[s]])
            
            
            start_bin= int((swim_startI[swim_inds1[s]]-period1_start[i])/600)
            end_bin  = int((swim_endI[swim_inds1[s]]-period1_start[i])/600)
            swim_trace[0,start_bin:end_bin,i] += swim_powers[swim_inds1[s]]/(end_bin-start_bin+1)
            
            
            
            
        swim_inds2=np.where((swim_startI>period2_start[i]) & (swim_startI<(period2_start[i]+120000)))[0]
        
        for s in range(len(swim_inds2)):
            
            swim_trials[1,i] += swim_powers[swim_inds2[s]]            
            highgain_swim.append(swim_powers[swim_inds2[s]])
            
            start_bin= int((swim_startI[swim_inds2[s]]-period2_start[i])/600)
            end_bin  = int((swim_endI[swim_inds2[s]]-period2_start[i])/600)
            swim_trace[1,start_bin:end_bin,i] += swim_powers[swim_inds2[s]]/(end_bin-start_bin+1)
            #print((start_bin,end_bin))
            
    
    swim_statistics.append((swim_trials,lowgain_swim,highgain_swim,
                            ttest_ind(swim_trials[0,:],swim_trials[1,:],alternative='greater')[1],ttest_ind(lowgain_swim,highgain_swim,alternative='greater')[1]))
    print(swim_statistics[-1][3])
    print(swim_statistics[-1][4])
    
    swim_trace=gaussian_filter1d(swim_trace,sigma=5,axis=1)
    swim_trace=swim_trace/np.mean(swim_trace[0,100:,:].flatten())
        
    summary_swim[:,:,dd,0]=np.mean(swim_trace,axis=2)
    summary_swim[:,:,dd,1]=np.std(swim_trace,axis=2)/np.sqrt(trials)
    
    
    
    
    for j in range(len(name_list)):
        
        #timeseries_raw=timeseries[j]
        inds_array=timeseries_inds_array[j]
        
        #%% optimizin pixels
        
        
        # activity_timeseries=timeseries_norm[j]
        # filter_array=np.zeros((activity_timeseries.shape[1],))

        
        # variance=activity_timeseries[1800:,:].std(axis=0)
        # variance_threshold = 0.035
        # filter_array[variance>variance_threshold]=2 ## sd filter
        
                
        # # plt.figure()
        # # plt.plot(np.arange(len(variance)),variance)
        
        
        
        # trial_activity=np.zeros((600,activity_timeseries.shape[1],2,trials))
        # trial_count=np.zeros((2,4))
    
        # for i in range(trials):
            
        #     closest_frame1=np.argmin(np.abs(frames-period1_start[i]))
            
        #     if (closest_frame1+600)<len(frames):
        #         trial_activity[:,:,0,i] = activity_timeseries[closest_frame1:closest_frame1+600,:]
        #         trial_count[0,:] +=1
        #     else:
        #         trial_activity[:,:,0,i] = np.nan
                
        #     closest_frame2=np.argmin(np.abs(frames-period2_start[i]))
            
        #     if (closest_frame2+600)<len(frames):
        #         trial_activity[:,:,1,i] = activity_timeseries[closest_frame2:closest_frame2+600,:]
        #         trial_count[1,:] +=1
        #     else:
        #         trial_activity[:,:,1,i] = np.nan
            
        # low_gain  = np.nanmean(trial_activity[150:,:,0,:],axis=0)
        # high_gain = np.nanmean(trial_activity[150:,:,1,:],axis=0)
        
        # stats=np.zeros((activity_timeseries.shape[1],1))
        # for i in range(activity_timeseries.shape[1]):
        #     r=ttest_ind(low_gain[i,~np.isnan(low_gain[i,:])],high_gain[i,~np.isnan(high_gain[i,:])])
        #     stats[i,0]=r[1]
        
        # filter_array[(stats[:,0]>0.05) & (filter_array!=2)]=1 ## task response filter
        
        # np.save(exp_dir+r'\\'+name_list[j]+'_ROI_filter_array.npy',filter_array)
        
        # # plt.close('all')
        # # pca = PCA(n_components=1)
        # # tmp=pca.fit_transform(activity_timeseries[1800:,:].T).flatten()
        # # a = pca.components_[0,:]
        # # likelihood=np.zeros((activity_timeseries.shape[1],2))
        # # for i in range(activity_timeseries.shape[1]):
        # #     r=pearsonr(activity_timeseries[1800:,i],a)
        # #     likelihood[i,0]=r[0]
        # #     likelihood[i,1]=r[1]
            
        # # inds=np.argsort(tmp[stat_filter])
        # # tmp_sorted=np.sort(tmp[stat_filter])
        # # plt.figure()
        # # plt.subplot(1,2,1)
        # # plt.imshow(trial_ave[:,stat_filter[inds],1].T, aspect=1)
        # # plt.subplot(1,2,2)
        # # plt.scatter(inds_array[0,stat_filter[inds]],inds_array[1,stat_filter[inds]],c=tmp_sorted,s=1)
        
        # # plt.figure()
        # # plt.plot(trial_ave[:,stat_filter,1].mean(axis=1))
        # # plt.figure()
        # # plt.plot(trial_ave[:,stat_filter,1].mean(axis=1))
        # # bbb
        
        
        
        #%% obtain final timecourse
        
        
        filter_array = np.load(exp_dir+r'\\'+name_list[j]+'_ROI_filter_array.npy')[()]
        
        # print(name_list[j]+r': filtered '+ str(timeseries_raw.shape[1]) +' into '+str(np.sum((filter_array==0).astype(np.int32)))+' pixels')

        # final_activity1=baseline_correction(timeseries_raw[:,filter_array==0].mean(axis=1),filter_size=15)      
        # final_activity2=baseline_correction(timeseries_raw[:,filter_array==0].mean(axis=1),filter_size=3)
           
        # np.save(exp_dir+r'\\'+name_list[j]+'_ROI_filter_array_final1.npy', final_activity1)
        # np.save(exp_dir+r'\\'+name_list[j]+'_ROI_filter_array_final2.npy', final_activity2)
        
        
        final_activity1 = np.load(exp_dir+r'\\'+name_list[j]+'_ROI_filter_array_final1.npy')[()]
        final_activity2 = np.load(exp_dir+r'\\'+name_list[j]+'_ROI_filter_array_final2.npy')[()]
        
        
        trial_activity=np.zeros((600,2,trials))
        trial_count=np.zeros((2,))
        
        for i in range(trials):
            
            closest_frame1=np.argmin(np.abs(frames-period1_start[i]))
            
            if (closest_frame1+600)<len(frames):
                trial_activity[:,0,i] = final_activity1[closest_frame1:closest_frame1+600]
                trial_count[0] +=1
            else:
                trial_activity[:,0,i] = np.nan
                
            closest_frame2=np.argmin(np.abs(frames-period2_start[i]))
            
            if (closest_frame2+600)<len(frames):
                trial_activity[:,1,i] = final_activity1[closest_frame2:closest_frame2+600]
                trial_count[1] +=1
            else:
                trial_activity[:,1,i] = np.nan
        
        trial_ave = np.nanmean(trial_activity,axis=2)
        trial_std = np.nanstd(trial_activity,axis=2)/np.sqrt(trial_count)[None,:,]
        
        summary_activity[:,:,j,dd,0] = trial_ave 
        summary_activity[:,:,j,dd,1] = trial_std 
        
            
                
        plt.figure(1)
        ax=plt.subplot(4,len(dir_list),(j*len(dir_list))+dd+1)
        plt.fill_between(np.arange(600)/30,trial_ave[:,0]-trial_std[:,0],trial_ave[:,0]+trial_std[:,0],alpha=0.2)
        plt.plot(np.arange(600)/30,trial_ave[:,0])
        plt.fill_between(np.arange(600,1200)/30,trial_ave[:,1]-trial_std[:,1],trial_ave[:,1]+trial_std[:,1],alpha=0.2)
        plt.plot(np.arange(600,1200)/30,trial_ave[:,1])
        plt.ylim(-0.005,0.01)
        if dd>0:
            ax.axes.yaxis.set_ticklabels([])
        if j<3:
            ax.axes.xaxis.set_ticklabels([])
            
        if dd==0:
            plt.ylabel(name_list[j])
        
        if j==0:
            plt.title(dir_list[dd][:-10].replace(r"\single",'\n single'))
            
        if swim_statistics[-1][3]<0.05:
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
                ax.spines[axis].set_color('green')
            
        
        plt.figure(2)
        ax=plt.subplot(4,len(dir_list),(j*len(dir_list))+dd+1)
        plt.scatter(inds_array[1,:],inds_array[0,:],c='k',marker=3)
        plt.scatter(inds_array[1,filter_array==0],inds_array[0,filter_array==0],c='r',marker=3)
        ax.axis('equal')
        ax.invert_yaxis()
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticklabels([])
        if dd==0:
            plt.ylabel(name_list[j])
        
        if j==0:
            plt.title(dir_list[dd][:-10].replace(r"\single",'\n single'))
            
        if swim_statistics[-1][3]<0.05:
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
                ax.spines[axis].set_color('green')
            
                    

        #%% plot example
        
        
        # start_time=period1_start[7] # exp 12 10 '05212023\single7_iSeroSnFR'
        
        # plt.figure(99)
        
        # if j==0:
        #     axt1=plt.subplot(3,1,1)
        #     plt.plot(np.arange(start_time-12000,start_time+30000,6)/6000, 10000*swimdata['fltCh1'][start_time-12000:start_time+30000:6])
        #     #plt.plot(np.arange(start_time-12000,start_time+30000,6)/6000, 10000*swimdata['fltCh2'][start_time-12000:start_time+30000:6])
            
        #     plt.subplot(3,1,2,sharex=axt1)
        #     plt.plot(np.arange(start_time-12000,start_time+30000,6)/6000, rawdata['velocity'][start_time-12000:start_time+30000:6])
        
        # if j<2:
        #     plt.subplot(3,1,3,sharex=axt1)
        #     plt.plot(frames/6000,final_activity[:len(frames)])
            
        # plt.xlim((start_time/6000)-2,(start_time/6000)+5)
        # plt.ylim(0,0.015)
        
        # if j==2:
        #     bbb
        
    
        #%%
        
    
        # start_time=period2_start[10] # exp 12 10 '05212023\single7_iSeroSnFR'
        # final_activity=baseline_correction(timeseries_raw[:,filter_array==0].mean(axis=1),filter_size=3)
        
        # plt.figure(100)
        
        # if j==0:
        #     axt2=plt.subplot(3,1,1)
        #     plt.plot(np.arange(start_time-12000,start_time+30000,6)/6000, swimdata['fltCh1'][start_time-12000:start_time+30000:6])
        #     plt.plot(np.arange(start_time-12000,start_time+30000,6)/6000, swimdata['fltCh2'][start_time-12000:start_time+30000:6])
            
        #     plt.subplot(3,1,2,sharex=axt2)
        #     plt.plot(np.arange(start_time-12000,start_time+30000,6)/6000, rawdata['velocity'][start_time-12000:start_time+30000:6])
        
        # if j<2:
        #     plt.subplot(3,1,3,sharex=axt2)
        #     plt.plot(frames/6000,final_activity[:len(frames)])
            
        # plt.xlim((start_time/6000)-2,(start_time/6000)+5)
        # plt.ylim(-0.02,0.01)
        
        # if j==2:
        #     bbb
        # continue
    
        #%%
                
        # plt.close(99) plot another example of adaptation
    
        # start_time=period2_start[11] # exp 12 11 '05212023\single8_iSeroSnFR'
        # #final_activity=baseline_correction(timeseries_raw[:,filter_array==0].mean(axis=1),filter_size=3)
        
        # plt.figure(98)
        
        # if j==1:
        #     axt=plt.subplot(2,1,1)
        #     plt.plot(np.arange(start_time-60000,start_time+60000,6)/6000, swimdata['fltCh1'][start_time-60000:start_time+60000:6])
        #     plt.plot(np.arange(start_time-60000,start_time+60000,6)/6000, swimdata['fltCh2'][start_time-60000:start_time+60000:6])
            
        #     plt.subplot(2,1,2,sharex=axt)
        #     plt.plot(np.arange(start_time-60000,start_time+60000,6)/6000, rawdata['velocity'][start_time-60000:start_time+60000:6])
        
        # if j==1:
            
        #     bbb
        #continue
    
    
    #%% swim_triggered average
    
        #final_activity=baseline_correction(timeseries_raw[:,filter_array==0].mean(axis=1),filter_size=3)
        
        averages=[]
        swim_averages=[]
        visual_averages=[]
        
        
        all_swim_index=np.where((swimdata['swimStartIndT']>frames[1800]) & (swimdata['swimStartIndT']<frames[-60]) & ((swimdata['swimEndIndT']/6-swimdata['swimStartIndT']/6)>50))[0][:-1]
    
        swim_start_T=swimdata['swimStartIndT']
        swim_end_T=swimdata['swimEndIndT']
        
        average_response=np.zeros((len(all_swim_index),66))
        delta_response=np.zeros((len(all_swim_index),2))
        
        for s in range(len(all_swim_index)):
            closest_frame=np.argmin(np.abs(frames-swim_start_T[all_swim_index[s]]))
            closest_frame_end=np.argmin(np.abs(frames-swim_end_T[all_swim_index[s]]))
            closest_frame_next=np.argmin(np.abs(frames-swim_start_T[all_swim_index[s]+1]))
            
            average_response[s,:] = final_activity2[closest_frame-6:closest_frame+60]-final_activity2[closest_frame-1:closest_frame+2].mean()
            delta_response[s,0]=final_activity2[closest_frame_next-3:closest_frame_next].mean()
            delta_response[s,1]=final_activity2[closest_frame_next-3:closest_frame_next].mean()-final_activity2[closest_frame-1:closest_frame+2].mean()
            
        swim_power_diff=(swim_powers[all_swim_index+1]-swim_powers[all_swim_index])/swim_powers[all_swim_index]
        
        # sensorimotor regression
        #regressor=np.vstack((zscore(np.log(swim_powers[all_swim_index])),zscore(np.log(back_velocity[all_swim_index])))).T
        
        gain_index=np.where((gain[all_swim_index]>0)& (swim_powers[all_swim_index]>0) & (back_velocity[all_swim_index,0]>0.01) & (back_velocity[all_swim_index,1]>0.01))[0]
        
        def fit_model(x,*y):
            
            return np.sum((x[0]*y[0]+x[1]*y[1]+x[2]*y[2]-y[3])**2)
        
        x0=[0,0,0]
        x_boundary=[[-1,1],[-1,1],[-1,1]]
            
            
    
        if j==0:
            
            #model.fit(zscore(back_velocity[all_swim_index])[:,None] ,zscore(delta_response[:,1]))
            
            
     
            fitting_info[dd,0] = spearmanr(back_velocity[all_swim_index[gain_index],0] ,delta_response[gain_index,1])[0]
            fitting_info[dd,1] = spearmanr(back_velocity[all_swim_index[gain_index],1] ,delta_response[gain_index,1])[0]
            
            # res = minimize(fit_model, x0, bounds=x_boundary, method='L-BFGS-B', args=((zscore(swim_powers[all_swim_index[gain_index]]),
            #                                                     zscore(back_velocity[all_swim_index[gain_index],0]),
            #                                                     zscore(back_velocity[all_swim_index[gain_index],1]),
            #                                                     delta_response[gain_index,1]
            #                                                     )))
            
            # regressor = np.vstack((zscore(swim_powers[all_swim_index[gain_index]]),
            #                                                     zscore(back_velocity[all_swim_index[gain_index],0]),
            #                                                     zscore(back_velocity[all_swim_index[gain_index],1]))).T
            
            
            
            #reg=LinearRegression().fit()
            
            #fitting_info[dd,2]=reg.coef_[0]
            #fitting_info[dd,3]=reg.coef_[1]
            #fitting_info[dd,4]=reg.coef_[2]
            
            
            # plt.figure()
            # plt.subplot(2,2,1).hist(np.log(swim_powers[all_swim_index[gain_index]]))
            # plt.subplot(2,2,2).hist(np.log(back_velocity[all_swim_index[gain_index],0]))
            # plt.subplot(2,2,3).hist(np.log(back_velocity[all_swim_index[gain_index],1]))
            # plt.subplot(2,2,4).hist(delta_response[gain_index,1])
            
            # bbb
            
            regressor = np.vstack((zscore(np.log(swim_powers[all_swim_index[gain_index]])),
                                                                zscore(np.log(back_velocity[all_swim_index[gain_index],0])),
                                                                zscore(np.log(back_velocity[all_swim_index[gain_index],1])))).T
            
            # Instantiate a gamma family model with the default link function.

            
            reg=LinearRegression().fit(regressor,zscore(delta_response[gain_index,1]))
            
            
            fitting_info[dd,2]=reg.coef_[0]
            fitting_info[dd,3]=reg.coef_[1]
            fitting_info[dd,4]=reg.coef_[2]
            
            # bbb
                        
        # sensorimotor prediction
        
        if j==0:
            tmp0=delta_response[:,0].copy()
            tmp1=delta_response[:,1].copy()
            #a=mutual_info_regression(delta_response[:,None],swim_power_diff)
            # prediction_info[dd,0,0]=mutual_info_regression(tmp0[:,None],swim_powers[all_swim_index+1])
            # prediction_info[dd,1,0]=mutual_info_regression(tmp1[:,None],swim_powers[all_swim_index+1])
            # prediction_info[dd,2,0]=mutual_info_regression(tmp0[:,None],swim_power_diff)
            # prediction_info[dd,3,0]=mutual_info_regression(tmp1[:,None],swim_power_diff)
            prediction_info[dd,0,0]=spearmanr(tmp0,swim_powers[all_swim_index+1])[0]
            prediction_info[dd,1,0]=spearmanr(tmp1,swim_powers[all_swim_index+1])[0]
            prediction_info[dd,2,0]=spearmanr(tmp0,swim_power_diff)[0]
            prediction_info[dd,3,0]=spearmanr(tmp1,swim_power_diff)[0]
            
            np.random.shuffle(tmp0)
            np.random.shuffle(tmp1)
            
            # prediction_info[dd,0,1]=mutual_info_regression(tmp0[:,None],swim_powers[all_swim_index+1])
            # prediction_info[dd,1,1]=mutual_info_regression(tmp1[:,None],swim_powers[all_swim_index+1])
            # prediction_info[dd,2,1]=mutual_info_regression(tmp0[:,None],swim_power_diff)
            # prediction_info[dd,3,1]=mutual_info_regression(tmp1[:,None],swim_power_diff)
            prediction_info[dd,0,1]=spearmanr(tmp0,swim_powers[all_swim_index+1])[0]
            prediction_info[dd,1,1]=spearmanr(tmp1,swim_powers[all_swim_index+1])[0]
            prediction_info[dd,2,1]=spearmanr(tmp0,swim_power_diff)[0]
            prediction_info[dd,3,1]=spearmanr(tmp1,swim_power_diff)[0]
            
        norm_m=average_response.flatten().mean()
        norm_v=average_response.flatten().std()
        

        for i in range(2):
            n=0
            swim_index=np.where((rawdata['envParam1'][swimdata['swimStartIndT']]==(i+1)) & (swimdata['swimStartIndT']>frames[1800]))[0]
            swim_start_T=swimdata['swimStartIndT'][swim_index]
            swim_end_T=swimdata['swimEndIndT'][swim_index]
            swim_durations = (swimdata['swimEndIndT'][swim_index]-swimdata['swimStartIndT'][swim_index])/6
            swim_intervals = (swimdata['swimStartIndT'][swim_index[:-1]+1]-swimdata['swimStartIndT'][swim_index[:-1]])/6
            
            average=np.zeros((len(swim_start_T),90))
            swim_average=np.zeros((len(swim_start_T),3000,2))
            visual_average=np.zeros((len(swim_start_T),3000))
            
            if i==0:
                period_start=np.where((rawdata['envParam1'][:-1]==2) & (rawdata['envParam1'][1:]==1))[0]
            else:
                period_start=np.where((rawdata['envParam1'][:-1]==1) & (rawdata['envParam1'][1:]==2))[0]
            
            for s in range(len(swim_start_T)-1):
                
                time_since = swim_start_T[s]-period_start[np.where(period_start<swim_start_T[s])[0][-1]]
                prec_swim=np.where( (swim_start_T>(swim_start_T[s]-time_since)) & (swim_start_T<=swim_start_T[s]))[0]
                
                if (swim_start_T[s]<(frames[-1]-12000)) and (swim_durations[s]>50) and (len(prec_swim)<=3):
                    
                    closest_frame=np.argmin(np.abs(frames-swim_start_T[s]))
                    closest_frame_next=np.argmin(np.abs(frames-swim_start_T[s+1]))
                    baseline=final_activity2[closest_frame-1:closest_frame+2].mean()
                    
                    average[s,:60] = final_activity2[closest_frame-30:closest_frame+30]-baseline
                    average[s,60:] = final_activity2[closest_frame_next-20:closest_frame_next+10]-baseline
                    #average[s,:] = final_activity2[closest_frame-30:closest_frame+60] #-final_activity2[closest_frame+12:closest_frame+15].mean()
                    swim_average[s,:,0] = swimdata['fltCh1'][swim_start_T[s]-6000:swim_start_T[s]+12000][::6]
                    swim_average[s,:,1] = swimdata['fltCh2'][swim_start_T[s]-6000:swim_start_T[s]+12000][::6]
                    visual_average[s,:] = rawdata['velocity'][swim_start_T[s]-6000:swim_start_T[s]+12000][::6]
                    n +=1
                else:
                    average[s,:] = np.nan
                    swim_average[s,:,:] = np.nan
                    visual_average[s,:] = np.nan
            
            average=(average-norm_m)/norm_v
                    
            
            
            averages.append((np.nanmean(average,axis=0),np.nanstd(average,axis=0)/np.sqrt(n)))
            
            summary_activity_fast[:,i,j,dd,0]=np.nanmean(average,axis=0)
            summary_activity_fast[:,i,j,dd,1]=np.nanstd(average,axis=0)
            
            swim_averages.append((np.nanmean(swim_average,axis=0),np.nanstd(swim_average,axis=0)/np.sqrt(n)))
            visual_averages.append((np.nanmean(visual_average,axis=0),np.nanstd(visual_average,axis=0)/np.sqrt(n)))
        
        
        plt.figure(3)
        
        
        ax=plt.subplot(4,len(dir_list),(j*len(dir_list))+dd+1)
        
        plt.fill_between(np.arange(-30,20)/30,averages[0][0][:50]-averages[0][1][:50],averages[0][0][:50]+averages[0][1][:50],alpha=0.5,color='b')
        plt.plot(np.arange(-30,20)/30,averages[0][0][:50],'b')
        plt.fill_between(np.arange(20)/30+1.166,averages[0][0][70:]-averages[0][1][70:],averages[0][0][70:]+averages[0][1][70:],alpha=0.5,color='b')
        plt.plot(np.arange(20)/30+1.166,averages[0][0][70:],'b')
        
        plt.fill_between(np.arange(-30,20)/30,averages[1][0][:50]-averages[1][1][:50],averages[1][0][:50]+averages[1][1][:50],alpha=0.5,color='m')
        plt.plot(np.arange(-30,20)/30,averages[1][0][:50],'m')
        plt.fill_between(np.arange(20)/30+1.166,averages[1][0][70:]-averages[1][1][70:],averages[1][0][70:]+averages[1][1][70:],alpha=0.5,color='m')
        plt.plot(np.arange(20)/30+1.166,averages[1][0][70:],'m')
        #plt.xlim(-0.2, 0.75)
        #plt.ylim(-0.0004,0.0012)
        plt.plot([1.5,1.5],[-2,2],'k--',alpha=0.5)
        plt.plot([0,0],[-2,2],'k--',alpha=0.5)
        #plt.ylim(-0.005,0.005)
        plt.ylim(-2,2)
        plt.xlim(-0.2,2)
        if dd>0:
            ax.axes.yaxis.set_ticklabels([])
        if j<3:
            ax.axes.xaxis.set_ticklabels([])

        if dd==0:
            plt.ylabel(name_list[j])
        if j==3:
            plt.xlabel('Time (s)')
        
        if j==0:
            plt.title(dir_list[dd][:-10].replace(r"\single",'\n single'))
            
        if swim_statistics[dd][3]<0.05:
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(2)
                ax.spines[axis].set_color('green')
            
            
        
        if j==0:
            
            plt.figure(4)
            
            
            ax=plt.subplot(2,len(dir_list),(0*len(dir_list))+dd+1)
            
            dc=np.argmax([swim_averages[0][0][:,0].max(),swim_averages[0][0][:,1].max()])
    
            plt.fill_between(np.arange(-1000,2000)/1000,swim_averages[0][0][:,dc]-swim_averages[0][1][:,dc],swim_averages[0][0][:,dc]+swim_averages[0][1][:,dc],alpha=0.2,color='b',linewidth=0)
            plt.plot(np.arange(-1000,2000)/1000,swim_averages[0][0][:,dc],color='b')
            
            
            plt.fill_between(np.arange(-1000,2000)/1000,swim_averages[1][0][:,dc]-swim_averages[1][1][:,dc],swim_averages[1][0][:,dc]+swim_averages[1][1][:,dc],alpha=0.2,color='m',linewidth=0)
            plt.plot(np.arange(-1000,2000)/1000,swim_averages[1][0][:,dc],color='m')
            
            
            plt.plot([0.75,0.75],[0,0.00004],'k--',alpha=0.5)
            plt.plot([0,0],[0,0.00004],'k--',alpha=0.5)
            
            plt.title(dir_list[dd][:-10].replace(r"\single",'\n single'))
            plt.xlim(-0.2,1)
            plt.ylim(0.00001,0.0001)
            
            ax.axes.xaxis.set_ticklabels([])
            
            if dd>0:
                ax.axes.yaxis.set_ticklabels([])
            else:
                plt.ylabel('swim signal')
                
            if swim_statistics[dd][3]<0.05:
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(2)
                    ax.spines[axis].set_color('green')
                
        
            
            ax=plt.subplot(2,len(dir_list),(1*len(dir_list))+dd+1)
            
            plt.fill_between(np.arange(-1000,2000)/1000,visual_averages[0][0]-visual_averages[0][1],visual_averages[0][0]+visual_averages[0][1],alpha=0.2,color='b',linewidth=0)
            plt.plot(np.arange(-1000,2000)/1000,visual_averages[0][0],color='b')
            
            plt.fill_between(np.arange(-1000,2000)/1000,visual_averages[1][0]-visual_averages[1][1],visual_averages[1][0]+visual_averages[1][1],alpha=0.2,color='m',linewidth=0)
            plt.plot(np.arange(-1000,2000)/1000,visual_averages[1][0],color='m')
            
            #plt.plot([0.75,0.75],[-8,2],'k--',alpha=0.5)
            plt.plot([0,0],[-12,3],'k--',alpha=0.5)
            plt.plot([0.75,0.75],[-12,3],'k--',alpha=0.5)
            plt.ylim(-12,3)
            plt.xlim(-0.2,1)
            
            if dd>0:
                ax.axes.yaxis.set_ticklabels([])
            else:
                plt.ylabel('visual velocity')
            
            plt.xlabel('Time (s)')
                
            if swim_statistics[dd][3]<0.05:
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(2)
                    ax.spines[axis].set_color('green')




plt.figure(1)
plt.tight_layout()
    
plt.figure(2)
plt.tight_layout()
    
plt.figure(3)
plt.tight_layout()
    
plt.figure(4)
plt.tight_layout()
    
swim_stat_array=np.array([swim_statistics[x][3] for x in range(len(dir_list))])


#%%


plt.figure(0,figsize=(6,6))
plt.title('adapting fish')


fish_ave = np.nanmean(summary_swim[:,:,swim_stat_array<0.05,0],axis=2)
fish_std = np.nanstd(summary_swim[:,:,swim_stat_array<0.05,0],axis=2)/np.sqrt(np.sum((swim_stat_array<0.05).astype(np.int32)))

titles=['HindbrainR','HindbrainL','TectumR','TectumL']

for j in range(len(dir_list)):
    if swim_stat_array[j]<0.05:
        plt.plot(np.arange(200)/10,summary_swim[0,:,j,0],'k',alpha=0.2,linewidth=1)
plt.fill_between(np.arange(200)/10,fish_ave[0,:]-fish_std[0,:],fish_ave[0,:]+fish_std[0,:],alpha=0.5)
plt.plot(np.arange(200)/10,fish_ave[0,:],linewidth=3)

for j in range(len(dir_list)):
    if swim_stat_array[j]<0.05:
        plt.plot(np.arange(200,400)/10,summary_swim[1,:,j,0],'k',alpha=0.2,linewidth=1)
plt.fill_between(np.arange(200,400)/10,fish_ave[1,:]-fish_std[1,:],fish_ave[1,:]+fish_std[1,:],alpha=0.5)
plt.plot(np.arange(200,400)/10,fish_ave[1,:],linewidth=3)

#plt.ylim(-0.002,0.013)
#plt.title(titles[i])
plt.xlabel('Time (s)')
plt.ylabel('dF/F0')

plt.tight_layout()
        
#%%

plt.figure(5,figsize=(6,6))
plt.title('adapting fish')


fish_ave = np.nanmean(summary_activity[:,:,:,swim_stat_array<0.05,0],axis=3)
fish_std = np.nanstd(summary_activity[:,:,:,swim_stat_array<0.05,0],axis=3)/np.sqrt(np.sum((swim_stat_array<0.05).astype(np.int32)))

titles=['HindbrainR','HindbrainL','TectumR','TectumL']
for i in np.arange(4):
    plt.subplot(2,2,i+1)
    
    for j in range(len(dir_list)):
        if swim_stat_array[j]<0.05:
            plt.plot(np.arange(600)/30,summary_activity[:,0,i,j,0],'k',alpha=0.2,linewidth=1)
    plt.fill_between(np.arange(600)/30,fish_ave[:,0,i]-fish_std[:,0,i],fish_ave[:,0,i]+fish_std[:,0,i],alpha=0.5)
    plt.plot(np.arange(600)/30,fish_ave[:,0,i],linewidth=3)
    
    for j in range(len(dir_list)):
        if swim_stat_array[j]<0.05:
            plt.plot(np.arange(600,1200)/30,summary_activity[:,1,i,j,0],'k',alpha=0.2,linewidth=1)
    plt.fill_between(np.arange(600,1200)/30,fish_ave[:,1,i]-fish_std[:,1,i],fish_ave[:,1,i]+fish_std[:,1,i],alpha=0.5)
    plt.plot(np.arange(600,1200)/30,fish_ave[:,1,i],linewidth=3)
    
    plt.ylim(-0.002,0.013)
    plt.title(titles[i])
    plt.xlabel('Time (s)')
    plt.ylabel('dF/F0')
    
plt.tight_layout()



#%%
plt.figure(6,figsize=(6,6))

fish_ave = np.nanmean(summary_activity_fast[:,:,:,swim_stat_array<0.05,0],axis=3)
fish_std = np.nanstd(summary_activity_fast[:,:,:,swim_stat_array<0.05,0],axis=3)/np.sqrt(np.sum((swim_stat_array<0.05).astype(np.int32)))

titles=['HindbrainR','HindbrainL','TectumR','TectumL']
for i in np.arange(4):
    plt.subplot(2,2,i+1)
    
    for j in range(len(dir_list)):
        if swim_stat_array[j]<0.05:
            plt.plot((np.arange(50)/30-1),summary_activity_fast[:50,0,i,j,0],'b',alpha=0.15,linewidth=1)
            plt.plot((np.arange(20)/30+1.166),summary_activity_fast[70:,0,i,j,0],'b',alpha=0.15,linewidth=1)
            
    plt.fill_between((np.arange(50)/30-1),fish_ave[:50,0,i]-fish_std[:50,0,i],fish_ave[:50,0,i]+fish_std[:50,0,i],color='b',alpha=0.3,linewidth=0)
    plt.fill_between((np.arange(20)/30+1.166),fish_ave[70:,0,i]-fish_std[70:,0,i],fish_ave[70:,0,i]+fish_std[70:,0,i],color='b',alpha=0.3,linewidth=0)
    plt.plot((np.arange(50)/30-1),fish_ave[:50,0,i],linewidth=3,color='b')
    plt.plot((np.arange(20)/30+1.166),fish_ave[70:,0,i],linewidth=3,color='b')
    
    
    for j in range(len(dir_list)):
        if swim_stat_array[j]<0.05:
            plt.plot((np.arange(50)/30-1),summary_activity_fast[:50,1,i,j,0],'m',alpha=0.15,linewidth=1)
            plt.plot((np.arange(20)/30+1.16),summary_activity_fast[70:,1,i,j,0],'m',alpha=0.15,linewidth=1)
            
    plt.fill_between((np.arange(50)/30-1),fish_ave[:50,1,i]-fish_std[:50,1,i],fish_ave[:50,1,i]+fish_std[:50,1,i],color='m',alpha=0.3,linewidth=0)
    plt.fill_between((np.arange(20)/30+1.166),fish_ave[70:,1,i]-fish_std[70:,1,i],fish_ave[70:,1,i]+fish_std[70:,1,i],color='m',alpha=0.3,linewidth=0)
    plt.plot((np.arange(50)/30-1),fish_ave[:50,1,i],linewidth=3,color='m')
    plt.plot((np.arange(20)/30+1.166),fish_ave[70:,1,i],linewidth=3,color='m')
    
    #plt.ylim(-0.005,0.005)
    plt.ylim(-2,2)
    plt.xlim(-0.2,2)
    
    plt.plot([1.5,1.5],[-2,2],'k--',alpha=0.5)
    plt.plot([0,0],[-2,2],'k--',alpha=0.5)
        
    plt.title(titles[i])
    plt.xlabel('Time (s)')
    plt.ylabel('dF/F0')
    
plt.tight_layout()

#%%

plt.figure(7,figsize=(6,3))
plt.title('adapting fish')



titles=['Hindbrain','Tectum']
for i in np.arange(2):
    plt.subplot(1,2,i+1)
    
    tmp_summary=summary_activity[:,:,(i*2):((i+1)*2),:,0].mean(axis=2)
    
    fish_ave = np.nanmean(tmp_summary[:,:,swim_stat_array<0.05],axis=2)
    fish_std = np.nanstd(tmp_summary[:,:,swim_stat_array<0.05],axis=2)/np.sqrt(np.sum((swim_stat_array<0.05).astype(np.int32)))
    
    for j in range(len(dir_list)):
        if swim_stat_array[j]<0.05:
            plt.plot(np.arange(600)/30,tmp_summary[:,0,j],'k',alpha=0.2,linewidth=1)
    plt.fill_between(np.arange(600)/30,fish_ave[:,0]-fish_std[:,0],fish_ave[:,0]+fish_std[:,0],alpha=0.5)
    plt.plot(np.arange(600)/30,fish_ave[:,0],linewidth=3)
    
    for j in range(len(dir_list)):
        if swim_stat_array[j]<0.05:
            plt.plot(np.arange(600,1200)/30,tmp_summary[:,1,j],'k',alpha=0.2,linewidth=1)
    plt.fill_between(np.arange(600,1200)/30,fish_ave[:,1]-fish_std[:,1],fish_ave[:,1]+fish_std[:,1],alpha=0.5)
    plt.plot(np.arange(600,1200)/30,fish_ave[:,1],linewidth=3)
    
    
    plt.ylim(-0.002,0.013)
    plt.title(titles[i]+' \n '+str(ttest_rel(tmp_summary[150:600,0,swim_stat_array<0.05].mean(axis=0),tmp_summary[150:600,1,swim_stat_array<0.05].mean(axis=0))[1]))
    plt.xlabel('Time (s)')
    plt.ylabel('dF/F0')
    
    
    
plt.tight_layout()



#%%
plt.figure(8,figsize=(5,6))


titles=['Hindbrain','Tectum']
for i in np.arange(2):
    
    
    plt.subplot(2,2,i*2+1)
    
    tmp_summary=summary_activity_fast[:,:,(i*2):((i+1)*2),:,0].mean(axis=2)
    
    fish_ave = np.nanmean(tmp_summary[:,:,swim_stat_array<0.05],axis=2)
    fish_std = np.nanstd(tmp_summary[:,:,swim_stat_array<0.05],axis=2)/np.sqrt(np.sum((swim_stat_array<0.05).astype(np.int32)))
        
    for j in range(len(dir_list)):
        if swim_stat_array[j]<0.05:
            plt.plot((np.arange(50)/30-1),tmp_summary[:50,0,j],'b',alpha=0.15,linewidth=1)
            plt.plot((np.arange(20)/30+1.166),tmp_summary[70:,0,j],'b',alpha=0.15,linewidth=1)
            
    plt.fill_between((np.arange(50)/30-1),fish_ave[:50,0]-fish_std[:50,0],fish_ave[:50,0]+fish_std[:50,0],color='b',alpha=0.3,linewidth=0)
    plt.fill_between((np.arange(20)/30+1.166),fish_ave[70:,0]-fish_std[70:,0],fish_ave[70:,0]+fish_std[70:,0],color='b',alpha=0.3,linewidth=0)
    plt.plot((np.arange(50)/30-1),fish_ave[:50,0],linewidth=3,color='b')
    plt.plot((np.arange(20)/30+1.166),fish_ave[70:,0],linewidth=3,color='b')
    
    
    for j in range(len(dir_list)):
        if swim_stat_array[j]<0.05:
            plt.plot((np.arange(50)/30-1),tmp_summary[:50,1,j],'m',alpha=0.15,linewidth=1)
            plt.plot((np.arange(20)/30+1.16),tmp_summary[70:,1,j],'m',alpha=0.15,linewidth=1)
            
    plt.fill_between((np.arange(50)/30-1),fish_ave[:50,1]-fish_std[:50,1],fish_ave[:50,1]+fish_std[:50,1],color='m',alpha=0.3,linewidth=0)
    plt.fill_between((np.arange(20)/30+1.166),fish_ave[70:,1]-fish_std[70:,1],fish_ave[70:,1]+fish_std[70:,1],color='m',alpha=0.3,linewidth=0)
    plt.plot((np.arange(50)/30-1),fish_ave[:50,1],linewidth=3,color='m')
    plt.plot((np.arange(20)/30+1.166),fish_ave[70:,1],linewidth=3,color='m')
    
    #plt.ylim(-0.005,0.005)
    plt.ylim(-2,2)
    plt.xlim(-0.2,2)
    
    plt.plot([1.5,1.5],[-2,2],'k--',alpha=0.5)
    plt.plot([0,0],[-2,2],'k--',alpha=0.5)
        
    plt.title(titles[i])
    plt.xlabel('Time (s)')
    plt.ylabel('dF/F0')
    
    plt.subplot(2,2,i*2+2)
    
    plt.scatter(np.arange(np.sum((swim_stat_array<0.05).astype(np.int32)))/100+1,tmp_summary[79:82,0,swim_stat_array<0.05].mean(axis=0),color='b',alpha=0.3,linewidth=0)
    plt.scatter(np.arange(np.sum((swim_stat_array<0.05).astype(np.int32)))/100+1.2,tmp_summary[79:82,1,swim_stat_array<0.05].mean(axis=0),color='m',alpha=0.3,linewidth=0)
    plt.errorbar(0+1.1,tmp_summary[79:82,0,swim_stat_array<0.05].mean(axis=0).mean(axis=0),yerr=tmp_summary[79:82,0,swim_stat_array<0.05].mean(axis=0).std(axis=0),fmt='bo')
    plt.errorbar(0.2+1.1,tmp_summary[79:82,1,swim_stat_array<0.05].mean(axis=0).mean(axis=0),yerr=tmp_summary[79:82,0,swim_stat_array<0.05].mean(axis=0).std(axis=0),fmt='mo')
    plt.ylim(-2,2)
    plt.title(titles[i]+' \n '+str(ttest_rel(tmp_summary[79:82,0,swim_stat_array<0.05].mean(axis=0),tmp_summary[79:82,1,swim_stat_array<0.05].mean(axis=0))[1]))
    plt.xlabel('Time (s)')
    plt.ylabel('dF/F0')
    
    
plt.tight_layout()

#%%


plt.figure(9,figsize=(8,4))

tmp1=prediction_info[:,:,0]

include_index=np.where(swim_stat_array<0.05)[0]

plt.subplot(1,3,1)
for i in range(4):
    plt.scatter(np.arange(len(include_index))/20+i,tmp1[include_index,i])
    plt.errorbar(i-0.1,tmp1[include_index,i].mean(),yerr=tmp1[include_index,i].std(),fmt='o')
plt.ylim(-0.4,0.4)
print(ttest_1samp(tmp1[include_index,1],0)[1])

tmp2=prediction_info[:,:,1]

plt.subplot(1,3,2)
for i in range(4):
    plt.scatter(np.arange(len(include_index))/20+i,tmp2[include_index,i])
    plt.errorbar(i-0.1,tmp2[include_index,i].mean(),yerr=tmp2[include_index,i].std(),fmt='o')
plt.ylim(-0.4,0.4)


plt.subplot(1,3,3)
for i in range(5):
    plt.scatter(np.arange(len(include_index))/20+i,fitting_info[include_index,i],alpha=0.2,linewidth=None)
    plt.errorbar(i-0.1,fitting_info[include_index,i].mean(),yerr=fitting_info[include_index,i].std(),fmt='o')
# for j in range(len(include_index)):
#     plt.plot([4+j/20,6+j/20],[fitting_info[include_index[j],4],fitting_info[include_index[j],6]],'k',alpha=0.2)
plt.ylim(-0.8,0.8)

print(ttest_1samp(fitting_info[include_index,4],0)[1])


        
    