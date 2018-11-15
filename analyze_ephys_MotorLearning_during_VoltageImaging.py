# -*- coding: utf-8 -*-

plt.close("all")
clear_all()


import ep
import imfunctions as im
import csv
from shutil import copy2

filelist = np.array([('','','')],dtype=[('base','S50'),('dir','S50'),('name1','S10')])


base_dir=r'U:\\Takashi'
with open("C:\\Users\\kawashimat\\Desktop\\MemoryExp_Filelist.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count==0:
            filelist[0]=(base_dir+'\\',row[0],row[1])
        else:
            filelist=np.append(filelist,np.array((base_dir+'\\',row[0],row[1]),dtype=filelist.dtype))
        line_count += 1

include_exp=np.zeros((len(filelist),))
power_dist=np.zeros((len(filelist),))

for f in range(len(filelist)):
    
    plt.close("all")
    fname=filelist[f]['base'].decode('UTF8')+filelist[f]['dir'].decode('UTF8')+'\\'+filelist[f]['name1'].decode('UTF8')
    print(fname)
    swimdir = fname+'\\swim\\'
    imgdir  = fname+'\\registered\\'
    datadir  = fname+'\\data\\'
    
    noise_thre  = 0.5    
    
    
    #%% calculate trail-by-trial power
    
    if not os.path.isfile(swimdir+"trial_inds.npy"):
        rawdata=np.load(swimdir+"rawdata.npy")[()]
        swimdata=np.load(swimdir+"swimdata.npy")[()]
        
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
        
        if os.path.isfile(swimdir+"ex_inds.npy"):
            ex_inds=np.load(swimdir+"ex_inds.npy")[()]
            for i in range(len(ex_inds)):
                trial_inds[trial_inds==(ex_inds[i]+1)]=0
            
        
        np.save(swimdir+"trial_inds",trial_inds);
        
        blocklist=np.zeros((2,reclen))
        blocklist[0,:] = trial_inds[:,None].T;
        blocklist[1,:] = [trial_inds==0][0][:,None].T;
        
        blockpowers=ep.calc_blockpowers(swimdata,blocklist)
        np.save(swimdir+"blockpowers",blockpowers);
                    
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
        
        plt.savefig(swimdir+"trial power.png")
        
        power_dist[f]=blockpowers[2,:].mean()
        
    else:
        
        blockpowers=np.load(swimdir+"blockpowers.npy")[()]
        power_dist[f]=blockpowers[2,:].mean()
        
    #%% calculate trial power
    
    if not os.path.isfile(swimdir+"swim_powers.npy"):
    #if True:
    
        rawdata     = np.load(swimdir+"rawdata.npy")[()]
        swimdata    = np.load(swimdir+"swimdata.npy")[()]
        trial_inds  = np.load(swimdir+"trial_inds.npy")[()]
        ntrials     = int(trial_inds.max())
        stimParam=rawdata['stimParam3']+(rawdata['stimParam4']-1)*4;
        #task_durations=[20,7,10,5,20,15,10,5,20,30,10,5]
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
                bursts     =swimdata['burstBothT'][swimStarts[s]:swimEnds[s]]
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
        swim_powers_training5s /= 5;

        np.save(swimdir+"swim_nums",swim_nums);
        np.save(swimdir+"swim_powers",swim_powers);
        np.save(swimdir+"fbout_powers",fbout_powers);
        np.save(swimdir+"swim_powers_training5s",swim_powers_training5s);
        np.save(swimdir+"swim_shapes",swim_shapes);
    else:
        swim_powers=np.load(swimdir+"swim_powers.npy")[()]
        swim_powers_training5s = np.load(swimdir+"swim_powers_training5s.npy")[()];
    
    #%% calculate frame-by-frame swimming
    
    if not os.path.isfile(swimdir+"frame_stimParams.npy"):
        
        
        rawdata     = np.load(swimdir+"rawdata.npy")[()]
        swimdata    = np.load(swimdir+"swimdata.npy")[()]
        trial_inds    = np.load(swimdir+"trial_inds.npy")[()]
        
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
                    
        np.save(swimdir+"frame_stimParams.npy",frame_stimParams);
        np.save(swimdir+"frame_tcourse.npy",frame_tcourse);
        np.save(swimdir+"trial_frame_inds.npy",trial_frame_inds);
        np.save(swimdir+"frame_inds.npy",frame_inds);
        np.save(swimdir+"frame_swim_tcourse.npy",frame_swim_tcourse);
    
    #%% copy files
    
      
    base=r"U:\\Ziqiang\\Takashi_DRN_project\\"
    folder1= filelist[f]['dir'].decode('UTF8')
    if folder1.startswith('0'):
        folder1=folder1.lstrip('0')

    folder2=filelist[f]['name1'].decode('UTF8')
        
    if not os.path.isdir(datadir):
        
        #os.mkdir(datadir)
        
        
        source_dir=base+folder1+'\\'+folder2+'\\Data'
        print('copying')
        copy2(source_dir+'\\Voltr_subvolt.npz',datadir)
        copy2(source_dir+'\\Voltr_spikes.npz',datadir)
        copy2(source_dir+'\\Voltr_raw.npz',datadir)
     
    #%% criteria 1 by power

        
    if power_dist[f]<1:
        include_exp[f]=1
        
        
    #%% criteria 2 by spontaneous swimming
    
    include_matrix=np.zeros((swim_powers.shape[0]))
    
    swim_powers_init     = swim_powers[:,[0,4,8]];
    swim_powers_training = swim_powers_training5s;
    swim_powers_delay    = swim_powers[:,[2,6,10]];
    delay_thre=(swim_powers_training.flatten().mean())/4
    
    include_matrix=[(swim_powers_delay<delay_thre)][0]  
    
    include_sum=include_matrix.sum(axis=0)
    np.save(swimdir+"include_matrix",include_matrix);
     
    if include_sum.min()<3:
        include_exp[f]=2
    
    if swim_powers_init.mean()<(swim_powers_training.mean()*2):
        include_exp[f]=3
        
    train_mean =  swim_powers_training5s.mean(axis=0)
    train_mean /= train_mean.mean()
    
    if (train_mean.min()< 0.4 or train_mean.max()>1.6):
        include_exp[f]=4
        
    
    
#%% criteria by swimming during OMR
    

include_inds=np.where(include_exp==0)[0]
filelist2=filelist[include_inds]
np.save(r'U:\\Takashi\DRN_MotorLearning_filelist1.npy',filelist)    
np.save(r'U:\\Takashi\DRN_MotorLearning_filelist2.npy',filelist2)    

swim_power_summaries    = np.zeros((len(include_inds),3,4))
fbout_power_summaries    = np.zeros((len(include_inds),3))

for f in range(len(include_inds)):
    
    swimdir=filelist[include_inds[f]]['base'].decode('UTF8')+filelist[include_inds[f]]['dir'].decode('UTF8')+'\\'+filelist[include_inds[f]]['name1'].decode('UTF8')+'\\swim\\'
    swim_shapes=np.load(swimdir+"swim_shapes.npy")[()];
    #include_matrix =np.load(swimdir+"include_matrix.npy")[()];
    swim_powers =np.load(swimdir+"swim_powers.npy")[()];
    fbout_powers =np.load(swimdir+"fbout_powers.npy")[()];
    swim_powers_training5s =np.load(swimdir+"swim_powers_training5s.npy")[()];
    
    plt.figure(1)
    plt.subplot(10,4,f+1).plot(swim_shapes.mean(axis=0));
    
    
    swim_powers_init     = swim_powers[:,[0,4,8]]
    #swim_powers_training = swim_powers[:,[1,5,9]]
    swim_powers_training = swim_powers_training5s
    swim_powers_delay    = swim_powers[:,[2,6,10]]
    swim_powers_test    = swim_powers[:,[3,7,11]]
    delay_thre=(swim_powers_training.flatten().mean())/4
    
    swim_power_summaries[f,:,0] = np.nanmean(swim_powers_init, axis=0)
    swim_power_summaries[f,:,1] = np.nanmean(swim_powers_training,axis=0)
    swim_power_summaries[f,:,2] = np.nanmean(swim_powers_delay,axis=0)
    swim_power_summaries[f,:,3] = np.nanmean(swim_powers_test,axis=0)
    
    div= swim_power_summaries[f,:,1].mean()
    
    swim_power_summaries[f,:,0] /= swim_power_summaries[f,:,0].mean()
    swim_power_summaries[f,:,1] /= div
    swim_power_summaries[f,:,2] /= div
    swim_power_summaries[f,:,3] /= swim_power_summaries[f,:,3].mean()
    
    fbout_power_test=fbout_powers[:,[3,7,11]].mean(axis=0)
    fbout_power_summaries[f,:]=fbout_power_test/fbout_power_test.mean()


plt.figure(2)
plt.subplot(1,5,1).plot(swim_power_summaries[:,:,0].T)
plt.subplot(1,5,1).errorbar(np.arange(3),np.nanmean(swim_power_summaries[:,:,0],axis=0),np.nanstd(swim_power_summaries[:,:,0],axis=0)/np.sqrt(len(include_inds)),linewidth=5)
plt.ylim([0,2])

plt.subplot(1,5,2).plot(swim_power_summaries[:,:,1].T)
plt.subplot(1,5,2).errorbar(np.arange(3),np.nanmean(swim_power_summaries[:,:,1],axis=0),np.nanstd(swim_power_summaries[:,:,1],axis=0)/np.sqrt(len(include_inds)),linewidth=5)
plt.ylim([0,2])

plt.subplot(1,5,3).plot(swim_power_summaries[:,:,2].T)
plt.subplot(1,5,3).plot(np.nanmean(swim_power_summaries[:,:,2],axis=0),linewidth=5)
plt.ylim([0,2])

plt.subplot(1,5,4).plot(swim_power_summaries[:,:,3].T)
plt.subplot(1,5,4).errorbar(np.arange(3),np.nanmean(swim_power_summaries[:,:,3],axis=0),np.nanstd(swim_power_summaries[:,:,3],axis=0)/np.sqrt(len(include_inds)),linewidth=5)
plt.ylim([0,2])

plt.subplot(1,5,5).plot(fbout_power_summaries.T)
plt.subplot(1,5,5).errorbar(np.arange(3),np.nanmean(fbout_power_summaries,axis=0),np.nanstd(fbout_power_summaries,axis=0)/np.sqrt(len(include_inds)),linewidth=5)
plt.ylim([0,2])
    