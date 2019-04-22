import numpy as np
import os, sys

noise_thre  = 0.5

def swim(folder, fish, rootDir, dat_folder):
    from fish_proc.utils.ep import process_swim
    swim_chFit = rootDir + f'{folder}/{fish}.10chFlt'
    save_folder = dat_folder + f'{folder}/{fish}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(save_folder+'/swim'):
        os.makedirs(save_folder+'/swim')
        print(f'checking file {folder}/{fish}')
        try:
            process_swim(swim_chFit, save_folder)
        except IOError:
            os.rmdir(save_folder+'/swim')
            print(f'Check existence of file {swim_chFit}')
    return None


def trial_swim_power(folder, fish, dir_folder):
    from fish_proc.utils import ep
    swimdir = dir_folder/f'{folder}/{fish}/swim'
    rawdata=np.load(swimdir/"rawdata.npy")[()]
    swimdata=np.load(swimdir/"swimdata.npy")[()]
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
        ex_inds=np.load(swimdir/"ex_inds.npy")[()]
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


def frame_swim_power(folder, fish, dir_folder):
    swimdir = dir_folder/f'{folder}/{fish}/swim'
    rawdata = np.load(swimdir/"rawdata.npy")[()]
    swimdata = np.load(swimdir/"swimdata.npy")[()]
    trial_inds = np.load(swimdir/"trial_inds.npy")[()]
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


def frame_swim_power_series(folder, fish, dir_folder):
    swimdir = dir_folder/f'{folder}/{fish}/swim'
    rawdata = np.load(swimdir/"rawdata.npy")[()]
    swimdata = np.load(swimdir/"swimdata.npy")[()]
    trial_inds = np.load(swimdir/"trial_inds.npy")[()]
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
