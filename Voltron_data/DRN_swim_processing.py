#!/groups/ahrens/home/weiz/anaconda3/envs/myenv/bin/python

import numpy as np
from pathlib import Path
import pandas as pd
from sys import platform
import matplotlib.pyplot as plt
import os, sys
from glob import glob
import fnmatch
import re


# Deal with path in windows and unix-like systems (solution only applied to python 3.0)
dat_xls_file = pd.read_csv('Voltron_Log_DRN_Exp.csv', index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
# using Path to handle switches filesystems
if platform == "linux" or platform == "linux2":
    dir_folder = Path('/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/')
elif platform == 'win32':
    dir_folder = Path('U:\\Takashi') # put folder for windows system
noise_thre  = 0.5


def findfiles(which, where='.'):
    '''Returns list of filenames from `where` path matched by 'which'
       shell pattern. Matching is case-insensitive.'''

    # TODO: recursive param with walk() filtering
    rule = re.compile(fnmatch.translate(which), re.IGNORECASE)
    return [name for name in os.listdir(where) if rule.match(name)]


def swim():
    '''
    Processing swim using TK's code
    '''
    dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'
    from fish_proc.utils.ep import process_swim
    for _, row in dat_xls_file.iterrows():
        if row['swim']:
            continue
        folder = row['folder']
        fish = row['fish']
        swim_chFit = row['rootDir'] + f'{folder}/{fish}.10chFlt'
        if not os.path.exists(swim_chFit):
            fish_alt = findfiles(fish+'.10chFlt', row['rootDir']+folder)
            if len(fish_alt)==1:
                swim_chFit = row['rootDir'] + f'{folder}/{fish_alt[0]}'
            else:
                swim_tmp = glob(row['rootDir']+f'{folder}/{fish[:7]}*.10chFlt')
                if len(swim_tmp) == 1:
                    swim_chFit = swim_tmp[0]
                else:
                    print(f'Check existence of file {swim_chFit}')
                    continue
        save_folder = dat_folder + f'{folder}/{fish}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(save_folder+'/swim'):
            os.makedirs(save_folder+'/swim')
            print(f'checking file {folder}/{fish}')
            try:
                print(f'Using matched file {swim_chFit}')
                process_swim(swim_chFit, save_folder)
            except IOError:
                os.rmdir(save_folder+'/swim')
                print(f'Check existence of file {swim_chFit}')
    return None


def trial_swim_power():
    '''
    Calculating trial-by-trial power
    '''
    from fish_proc.utils import ep
    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        swimdir = dir_folder/f'{folder}/{fish}/swim'
        if os.path.isfile(swimdir/"trial_inds.npy"):
            continue
        if not os.path.isfile(swimdir/"rawdata.npy"):
            print(f'Preprocessing is not done, skip trial_swim_power at {folder}_{fish}')
            continue
        rawdata=np.load(swimdir/"rawdata.npy", allow_pickle=True)[()]
        swimdata=np.load(swimdir/"swimdata.npy", allow_pickle=True)[()]
        reclen=len(swimdata['fltCh1'])
        trial_inds=np.zeros((reclen,))
        rawdata['stimParam3'][:50]=0
        rawdata['stimParam4'][:50]=0
        stimParam=rawdata['stimParam3']+(rawdata['stimParam4']-1)*4
        max_stimParam=stimParam.max()
        end_stimParam=np.where(np.diff((stimParam==max_stimParam).astype('int'))==-1)[0]+1
        trial_inds[:end_stimParam[0]]=1
        for i in range(1,len(end_stimParam)):
            trial_inds[end_stimParam[i-1]:end_stimParam[i]]=i+1
        trial_inds[end_stimParam[-1]:]=len(end_stimParam)+1
        trial_inds[trial_inds==trial_inds.max()]=0
        if os.path.isfile(swimdir/"ex_inds.npy"):
            ex_inds=np.load(swimdir/"ex_inds.npy", allow_pickle=True)[()]
            for i in range(len(ex_inds)):
                trial_inds[trial_inds==(ex_inds[i]+1)]=0
        np.save(swimdir/"trial_inds",trial_inds);
        blocklist=np.zeros((2,reclen))
        blocklist[0,:] = trial_inds[:,None].T;
        blocklist[1,:] = [trial_inds==0][0][:,None].T;
        blockpowers=ep.calc_blockpowers(swimdata,blocklist)
        np.save(swimdir/"blockpowers",blockpowers);
        totblock=(blocklist.max()).astype('i4');
        plt.figure(1,figsize=(16,5))
        plt.subplot(131)
        plt.bar(np.arange(1,totblock+1,1),blockpowers[0,]);
        plt.title('Left')
        plt.xlim(0,totblock+1)
        plt.subplot(132)
        plt.bar(np.arange(1,totblock+1,1),blockpowers[1,]);
        plt.title('Right')
        plt.xlim(0,totblock+1)
        plt.subplot(133)
        plt.bar(np.arange(1,totblock+1,1),blockpowers[2,]);
        plt.title('Sum')
        plt.xlim(0,totblock+1)
        plt.savefig(swimdir/"trial power.png")


def trial_power():
    '''
    Calculating trial power
    '''
    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        swimdir = dir_folder/f'{folder}/{fish}/swim'
        if os.path.isfile(swimdir/"swim_powers.npy"):
            continue
        if not row['task']=='Raphe memory task':
            continue
        if not os.path.isfile(swimdir/"rawdata.npy"):
            print(f'Preprocessing is not done, skip trial_power at {folder}_{fish}')
            continue
        rawdata     = np.load(swimdir/"rawdata.npy", allow_pickle=True)[()]
        swimdata    = np.load(swimdir/"swimdata.npy", allow_pickle=True)[()]
        trial_inds  = np.load(swimdir/"trial_inds.npy", allow_pickle=True)[()]
        ntrials     = int(trial_inds.max())
        stimParam=rawdata['stimParam3']+(rawdata['stimParam4']-1)*4;
        # 4 trials per recording
        # 3 session per trial
        # 4 epoch per session
        task_durations=[20,7,8,5,20,15,8,5,20,30,8,5]
        trial_index=np.zeros((ntrials,12))
        for i in range(ntrials):
            span=np.where(trial_inds==(i+1))[0]
            for j in range(12):
                start=np.where(stimParam[span]==(j+1))[0][0]
                trial_index[i,j] =  span[0]+start
        swim_powers=np.zeros((ntrials,12))
        fbout_powers=np.zeros((ntrials,12))
        swim_powers_training5s=np.zeros((ntrials,3))
        swim_shapes=np.zeros((ntrials, 157*6000+600))
        swim_nums=np.zeros((ntrials,12));
        swimStarts=swimdata['swimStartIndT']
        swimEnds =swimdata['swimEndIndT']
        swimDurs =swimEnds-swimStarts
        bursts=swimdata['burstBothT']
        if isinstance(swimStarts,np.ndarray):
            for s in range(1,len(swimStarts)):
                ch1_partial=rawdata['ch1'][swimStarts[s]:swimEnds[s]]
                ch2_partial=rawdata['ch2'][swimStarts[s]:swimEnds[s]]
                bursts =swimdata['burstBothT'][swimStarts[s]:swimEnds[s]]
                ch1_max=(np.abs(ch1_partial-np.median(ch1_partial))).max()
                ch2_max=(np.abs(ch2_partial-np.median(ch2_partial))).max()
                trial=int(trial_inds[swimStarts[s]])
                if (max(ch1_max,ch2_max)<noise_thre and swimDurs[s]>60 and trial>0):
                    task_period=int(stimParam[swimStarts[s]])
                    p1=(swimdata['fltCh1']-swimdata['back1'])[swimStarts[s]:swimEnds[s]].sum()
                    p2=(swimdata['fltCh2']-swimdata['back2'])[swimStarts[s]:swimEnds[s]].sum()
                    p_sum=p1*(p1>0)+p2*(p2>0)
                    swim_shapes[trial-1, int(swimStarts[s]-trial_index[trial-1,0]):int(swimEnds[s]-trial_index[trial-1,0])]=p_sum/swimDurs[s]*1000
                    swim_nums[trial-1,task_period-1]+=1
                    if swim_nums[trial-1,task_period-1]==1:
                        fbout_powers[trial-1,task_period-1] += p_sum;
                    # dismiss first 2 sec of the delay period
                    if (np.mod(task_period,4)==3):
                        if (swimStarts[s]-trial_index[trial-1,task_period-1])>12000:
                            swim_powers[trial-1,task_period-1]+=p_sum
                    else:
                        swim_powers[trial-1,task_period-1]+=p_sum
                    # separately calculate swim power in the last 5s of training period
                    if (np.mod(task_period,4)==2 and (trial_index[trial-1,task_period]-swimStarts[s])<30000):
                        swim_powers_training5s[trial-1,int((task_period-1)/4)]+=p_sum
        swim_powers /= np.array(task_durations)[:,None].T
        swim_powers_training5s /= 5
        np.save(swimdir/"swim_nums",swim_nums)
        np.save(swimdir/"swim_powers",swim_powers)
        np.save(swimdir/"fbout_powers",fbout_powers)
        np.save(swimdir/"swim_powers_training5s",swim_powers_training5s)
        np.save(swimdir/"swim_shapes",swim_shapes)


def frame_swim_power():
    '''
    Calculating frame-by-frame power
    '''
    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        swimdir = dir_folder/f'{folder}/{fish}/swim'
        if os.path.isfile(swimdir/"frame_stimParams.npy"):
            continue
        if not os.path.isfile(swimdir/"rawdata.npy"):
            print(f'Preprocessing is not done, skip frame_swim_power at {folder}_{fish}')
            continue
        rawdata = np.load(swimdir/"rawdata.npy", allow_pickle=True)[()]
        swimdata = np.load(swimdir/"swimdata.npy", allow_pickle=True)[()]
        trial_inds = np.load(swimdir/"trial_inds.npy", allow_pickle=True)[()]
        reclen=len(swimdata['fltCh1'])
        frame=np.where(np.diff((rawdata['ch3']>3).astype('int'))==1)[0]
        if 'stimParam6' in rawdata.keys():
            frame_stimParams=np.zeros((6,len(frame)))
        else:
            frame_stimParams=np.zeros((5,len(frame)))
        frame_stimParams[0,:]=rawdata['stimParam1'][frame]
        frame_stimParams[2,:]=rawdata['stimParam3'][frame]
        frame_stimParams[3,:]=rawdata['stimParam4'][frame]
        frame_stimParams[4,:]=rawdata['stimParam5'][frame]
        if 'stimParam6' in rawdata.keys():
            frame_stimParams[5,:]=rawdata['stimParam6'][frame]
        frame_stimParams[:,:3]=frame_stimParams[:,3:6];
        trial_frame_inds=trial_inds[frame]
        frame_tcourse=np.zeros((reclen,))
        frame_inds=np.zeros((len(frame)-1,2))
        for t in range(len(frame)-1):
            frame_tcourse[frame[t]:frame[t+1]]=t
            frame_inds[t,0]=frame[t]
            frame_inds[t,1]=frame[t+1]-1
        swimStarts=swimdata['swimStartIndT']
        swimEnds =swimdata['swimEndIndT']
        swimDurs =swimEnds-swimStarts
        frame_swim_tcourse=np.zeros((3,len(frame)))
        if not np.isscalar(swimStarts):
            for s in range(1,len(swimStarts)):
                startI  = swimStarts[s]
                endI    = swimEnds[s]
                fstart = int(frame_tcourse[startI])
                fend   = int(frame_tcourse[endI])
                ch1_partial=rawdata['ch1'][swimStarts[s]:swimEnds[s]]
                ch2_partial=rawdata['ch2'][swimStarts[s]:swimEnds[s]]
                bursts     =swimdata['burstBothT'][swimStarts[s]:swimEnds[s]]
                ch1_max=(np.abs(ch1_partial-np.median(ch1_partial))).max()
                ch2_max=(np.abs(ch2_partial-np.median(ch2_partial))).max()
                if (max(ch1_max,ch2_max)<noise_thre and swimDurs[s]>60 and fstart>0 and fend>0):
                    swim_pow1=max(swimdata['fltCh1'][startI:endI].sum()-swimdata['back1'][startI:endI].sum(),0)
                    swim_pow2=max(swimdata['fltCh2'][startI:endI].sum()-swimdata['back2'][startI:endI].sum(),0)
                    swim_pow_sum=swim_pow1+swim_pow2
                    frame_swim_tcourse[0,fstart:fend]=swim_pow_sum/(fend-fstart)
                    frame_swim_tcourse[1,fstart:fend]=swim_pow1/(fend-fstart)
                    frame_swim_tcourse[2,fstart:fend]=swim_pow2/(fend-fstart)
        np.save(swimdir/"frame_stimParams.npy",frame_stimParams)
        np.save(swimdir/"frame_tcourse.npy",frame_tcourse)
        np.save(swimdir/"trial_frame_inds.npy",trial_frame_inds)
        np.save(swimdir/"frame_inds.npy",frame_inds);
        np.save(swimdir/"frame_swim_tcourse.npy",frame_swim_tcourse)


def match_swim_frame(swim_pow, startI, endI, fstart, fend, frame_tcourse):
    d_swim_pow = np.zeros(fend-fstart)
    for n_t in range(fend-fstart):
        d_swim_pow[n_t] = swim_pow[frame_tcourse == (fstart + n_t)].mean()
    return d_swim_pow


def frame_swim_power_series():
    '''
    Calculating frame-by-frame power
    '''
    for index, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        swimdir = dir_folder/f'{folder}/{fish}/swim'
        # if os.path.isfile(swimdir/"frame_swim_tcourse_series.npy"):
        #     continue
        if not os.path.isfile(swimdir/"rawdata.npy"):
            print(f'Preprocessing is not done, skip frame_swim_power_series at {folder}_{fish}')
            continue
        if os.path.isfile(swimdir/"frame_swim_tcourse_series.npy"):
            continue
        rawdata = np.load(swimdir/"rawdata.npy", allow_pickle=True)[()]
        swimdata = np.load(swimdir/"swimdata.npy", allow_pickle=True)[()]
        trial_inds = np.load(swimdir/"trial_inds.npy", allow_pickle=True)[()]
        reclen=len(swimdata['fltCh1'])
        frame=np.where(np.diff((rawdata['ch3']>3).astype('int'))==1)[0]
        frame_stimParams=np.zeros((5,len(frame)))
        frame_stimParams[0,:]=rawdata['stimParam1'][frame]
        frame_stimParams[2,:]=rawdata['stimParam3'][frame]
        frame_stimParams[3,:]=rawdata['stimParam4'][frame]
        frame_stimParams[4,:]=rawdata['stimParam5'][frame]
        frame_stimParams[:,:3]=frame_stimParams[:,3:6];
        trial_frame_inds=trial_inds[frame]
        frame_tcourse=np.zeros((reclen,))
        frame_inds=np.zeros((len(frame)-1,2))
        for t in range(len(frame)-1):
            frame_tcourse[frame[t]:frame[t+1]]=t
            frame_inds[t,0]=frame[t]
            frame_inds[t,1]=frame[t+1]-1
        swimStarts=swimdata['swimStartIndT']
        swimEnds =swimdata['swimEndIndT']
        swimDurs =swimEnds-swimStarts
        frame_swim_tcourse=np.zeros((3,len(frame)))
        if not np.isscalar(swimStarts):
            for s in range(1,len(swimStarts)):
                startI  = swimStarts[s]
                endI    = swimEnds[s]
                fstart = int(frame_tcourse[startI])
                fend   = int(frame_tcourse[endI])
                ch1_partial=rawdata['ch1'][swimStarts[s]:swimEnds[s]]
                ch2_partial=rawdata['ch2'][swimStarts[s]:swimEnds[s]]
                bursts     =swimdata['burstBothT'][swimStarts[s]:swimEnds[s]]
                ch1_max=(np.abs(ch1_partial-np.median(ch1_partial))).max()
                ch2_max=(np.abs(ch2_partial-np.median(ch2_partial))).max()
                if (max(ch1_max,ch2_max)<noise_thre and swimDurs[s]>60 and fstart>0 and fend>0):
                    swim_pow1=np.clip(swimdata['fltCh1'][startI:endI]-swimdata['back1'][startI:endI], 0, None)
                    swim_pow2=np.clip(swimdata['fltCh2'][startI:endI]-swimdata['back2'][startI:endI], 0, None)
                    swim_pow_sum=swim_pow1+swim_pow2
                    frame_tcourse_ = frame_tcourse[startI:endI]
                    frame_swim_tcourse[0,fstart:fend]=match_swim_frame(swim_pow_sum, startI, endI, fstart, fend, frame_tcourse_)
                    frame_swim_tcourse[1,fstart:fend]=match_swim_frame(swim_pow1, startI, endI, fstart, fend, frame_tcourse_)
                    frame_swim_tcourse[2,fstart:fend]=match_swim_frame(swim_pow2, startI, endI, fstart, fend, frame_tcourse_)
        np.save(swimdir/"frame_swim_tcourse_series.npy",frame_swim_tcourse)


if __name__ == '__main__':
    swim()
    trial_swim_power()
    trial_power()
    frame_swim_power()
    frame_swim_power_series()
