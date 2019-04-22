import ep
import os
import imfunctions as im
import array as ar
import struct as st
import scipy.io as sio
import pandas as pd
from sklearn.decomposition import NMF


pathname=r"D:\Takashi\SPIM_newPC\04182019";
fname="Fish1-3-delay"

noise_thre=0.5
frame_rate=30

img_dir=pathname+"\\"+fname+"\\Registered\\"
swm_dir=pathname+"\\"+fname+"\\swim\\"


#%%
## calculate frame swim spower
#####################################

if not os.path.isfile(swm_dir+"frame_swim_tcourse.npy"):    
    
    rawdata=np.load(swm_dir+'rawdata.npy')[()]
    swimdata=np.load(swm_dir+'swimdata.npy')[()]
    
    reclen=len(swimdata['fltCh1'])
    
    frame=np.where(np.diff((rawdata['ch3']>3).astype('int'))==1)[0]
    frame=frame[:-1]
    
    frame_tcourse=np.zeros((reclen,))
    for t in range(len(frame)-1):
        frame_tcourse[int((frame[t]+frame[t+1])/2)]=(t+1)
        
        
    frame_stimParams=np.zeros((5,len(frame)))
    frame_stimParams[0,:]=rawdata['stimParam1'][frame]
    frame_stimParams[2,:]=rawdata['stimParam3'][frame]
    frame_stimParams[3,:]=rawdata['stimParam4'][frame]
    frame_stimParams[4,:]=rawdata['stimParam5'][frame]
    
    
    reclen=len(swimdata['fltCh1'])
    
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
        
        ch1_partial=rawdata['ch1'][swimStarts[s]:swimEnds[s]]
        ch2_partial=rawdata['ch2'][swimStarts[s]:swimEnds[s]]
        bursts     =swimdata['burstBothT'][swimStarts[s]:swimEnds[s]]
        ch1_max=(np.abs(ch1_partial-np.median(ch1_partial))).max()
        ch2_max=(np.abs(ch2_partial-np.median(ch2_partial))).max()
        
        if (max(ch1_max,ch2_max)<noise_thre and swimDurs[s]>60):
        
            swim_pow1=max(swimdata['fltCh1'][startI:endI].sum()-swimdata['back1'][startI:endI].sum(),0)
            swim_pow2=max(swimdata['fltCh2'][startI:endI].sum()-swimdata['back2'][startI:endI].sum(),0)
            swim_pow_sum=swim_pow1+swim_pow2
            fstart = int(frame_tcourse[startI])
            fend   = int(frame_tcourse[endI])
            dur=max(1,fend-fstart)
            frame_swim_tcourse[0,fstart:fend]=swim_pow_sum/dur
            
    frame_swim_tcourse[1,:]=swimdata['fltCh1'][frame]
    frame_swim_tcourse[2,:]=swimdata['fltCh2'][frame]
                
    np.save(swm_dir+"frame_swim_tcourse.npy",frame_swim_tcourse);
    np.save(swm_dir+"frame_stimParams.npy",frame_stimParams);
    

#%%
## calculate correlation with swimming
#####################################

frame_stimParams    = np.load(swm_dir+'\\frame_stimParams.npy')[()];
frame_swim_tcourse  = np.load(swm_dir+"\\frame_swim_tcourse.npy")[()];
dFF                 = np.load(img_dir+'\\dFF_sub.npy')[()]
ave                 = imread(img_dir+'\\ave.tif')
stack               = np.load(img_dir+'\\stack_sub.npy')[()]
frame_stimParams    = np.load(swm_dir+'\\frame_stimParams.npy')[()]


frame_len=frame_stimParams.shape[1]

dFF=dFF[:frame_len,:,:]
dim=dFF.shape

ave_sub=stack.mean(axis=0).reshape((1,dim[1]*dim[2]),order='F')
include_pix=np.where(ave_sub>150)[1]

dFF_square=dFF.reshape((dim[0],dim[1]*dim[2]),order='F')
dFF_square_norm=(dFF_square-dFF_square.mean(axis=0)[None,:])/dFF_square.std(axis=0)[None,:]

swim_start=np.where(np.diff((frame_swim_tcourse[0,:]>0).astype('int'))==1)[0]
swim_end=np.where(np.diff((frame_swim_tcourse[0,:]>0).astype('int'))==-1)[0]

swim_000ms = swim_start[np.where(frame_stimParams[2,swim_start]==1)[0]];
swim_200ms = swim_start[np.where(frame_stimParams[2,swim_start]==2)[0]];
swim_400ms = swim_start[np.where(frame_stimParams[2,swim_start]==3)[0]];
swim_000ms = swim_000ms[((swim_000ms>15) & (swim_000ms<(dFF_square.shape[0]-30)))]
swim_200ms = swim_200ms[((swim_200ms>15) & (swim_200ms<(dFF_square.shape[0]-30)))]
swim_400ms = swim_400ms[((swim_400ms>15) & (swim_400ms<(dFF_square.shape[0]-30)))]
swim_000ms = swim_000ms[:-1][(np.diff(swim_000ms)>30) ]
swim_200ms = swim_200ms[:-1][(np.diff(swim_200ms)>30) ]
swim_400ms = swim_400ms[:-1][(np.diff(swim_400ms)>30) ]


swim_000ms_matrix  = np.zeros((45,len(include_pix)))
swim_000ms_ave  = np.zeros((45,2))
for s in swim_000ms:
    swim_000ms_matrix   += dFF_square[(s-15):(s+30),include_pix]-dFF_square[(s-15):s,include_pix].mean(axis=0)[None,:]
    swim_000ms_ave[:,0] += frame_swim_tcourse[1,(s-15):(s+30)]
    swim_000ms_ave[:,1] += -frame_stimParams[0,(s-15):(s+30)]
swim_000ms_matrix /= len(swim_000ms)
swim_000ms_ave    /= len(swim_000ms)

swim_200ms_matrix  = np.zeros((45,len(include_pix)))
swim_200ms_ave  = np.zeros((45,2))
for s in swim_200ms:
    swim_200ms_matrix   += dFF_square[(s-15):(s+30),include_pix]-dFF_square[(s-15):s,include_pix].mean(axis=0)[None,:]
    swim_200ms_ave[:,0] += frame_swim_tcourse[1,(s-15):(s+30)]
    swim_200ms_ave[:,1] += -frame_stimParams[0,(s-15):(s+30)]
swim_200ms_matrix /= len(swim_200ms)
swim_200ms_ave    /= len(swim_200ms)

swim_400ms_matrix  = np.zeros((45,len(include_pix)))
swim_400ms_ave  = np.zeros((45,2))
for s in swim_400ms:
    swim_400ms_matrix   += dFF_square[(s-15):(s+30),include_pix]-dFF_square[(s-15):s,include_pix].mean(axis=0)[None,:]
    swim_400ms_ave[:,0] += frame_swim_tcourse[1,(s-15):(s+30)]
    swim_400ms_ave[:,1] += -frame_stimParams[0,(s-15):(s+30)]
swim_400ms_matrix /= len(swim_400ms)
swim_400ms_ave    /= len(swim_400ms)


plt.figure(1,figsize=(3,5))
plt.subplot(2,1,1).plot(np.arange(-15,30)/30, swim_000ms_ave[:,0]*10000,'k')
plt.subplot(2,1,1).plot(np.arange(-15,30)/30, swim_200ms_ave[:,0]*10000,'r')
plt.subplot(2,1,1).plot(np.arange(-15,30)/30, swim_400ms_ave[:,0]*10000,'b')
plt.title('event num: 0ms '+str(len(swim_000ms))+', 200ms '+str(len(swim_200ms))+', 400ms '+str(len(swim_400ms)))
plt.subplot(2,1,2).plot(np.arange(-15,30)/30, swim_000ms_ave[:,1]*10000,'k')
plt.subplot(2,1,2).plot(np.arange(-15,30)/30, swim_200ms_ave[:,1]*10000,'r')
plt.subplot(2,1,2).plot(np.arange(-15,30)/30, swim_400ms_ave[:,1]*10000,'b')

plt.figure(2,figsize=(4,4))
plt.subplot(3,1,1).imshow(swim_000ms_matrix,vmax=0.01,vmin=-0.005,aspect='auto', extent=(0,swim_000ms_matrix.shape[1], 1.0,-0.5))
plt.subplot(3,1,2).imshow(swim_200ms_matrix,vmax=0.01,vmin=-0.005,aspect='auto', extent=(0,swim_200ms_matrix.shape[1], 1.0,-0.5))
plt.subplot(3,1,3).imshow(swim_400ms_matrix,vmax=0.01,vmin=-0.005,aspect='auto', extent=(0,swim_400ms_matrix.shape[1], 1.0,-0.5))


#plt.figure(3,figsize=(3,6))
#plt.subplot(4,1,1).imshow(ave,aspect='auto')
#plt.subplot(4,1,2).imshow(swim_000ms_matrix[15:24,:].mean(axis=0).reshape((dim[1],dim[2]),order='F'),vmax=0.01,vmin=-0.005,aspect='auto')
#plt.subplot(4,1,3).imshow(swim_200ms_matrix[15:24,:].mean(axis=0).reshape((dim[1],dim[2]),order='F'),vmax=0.01,vmin=-0.005,aspect='auto')
#plt.subplot(4,1,4).imshow(swim_400ms_matrix[15:24,:].mean(axis=0).reshape((dim[1],dim[2]),order='F'),vmax=0.01,vmin=-0.005,aspect='auto')


plt.figure(4,figsize=(3,3))
plt.plot(np.arange(-15,30)/30,swim_000ms_matrix.mean(axis=1),'k')
plt.plot(np.arange(-15,30)/30,swim_200ms_matrix.mean(axis=1),'r')
plt.plot(np.arange(-15,30)/30,swim_400ms_matrix.mean(axis=1),'b')


bbb

## calculate fluorescence change between gains
## contaminated by blood shadows

#frame_stimParams=np.load(swm_dir+'\\frame_stimParams.npy')[()]
#plt.figure(2)
#plt.plot(frame_stimParams[2,:])
#
#low_frames  = np.where(frame_stimParams[2,:]==1)[0]
#high_frames = np.where(frame_stimParams[2,:]==2)[0]
#
#low_average  = dFF[low_frames ,:,:].mean(axis=0)
#high_average = dFF[high_frames,:,:].mean(axis=0)
#
#low_area =(low_average>high_average)
#high_area=(high_average>low_average)
#
#plt.figure(1)
#plt.subplot(1,2,1).imshow(low_area)
#plt.subplot(1,2,2).imshow(high_area)



## calculate NMF on a subsampled data 
## (difficult because of the blood shadow)
#####################################

#maxv=dFF_square.max()
#
#model = NMF(n_components=10, init='nndsvd')
#
#W = model.fit_transform(dFF_square)
#H = model.components_
#
#plt.figure(2)
#plt.plot(W)
##plt.plot(dFF_square.mean(axis=1))
#
#plt.figure(3)
#for i in range(5):
#    weight_matrix=H[i,:].reshape((dim[1],dim[2]),order='F')
#    plt.subplot(1,5,i+1).imshow(weight_matrix)


dFF = np.load(img_dir+'\\dFF.npy')[()]
dim=dFF.shape

dFF_square=dFF.reshape((dim[0],dim[1]*dim[2]),order='F')
dFF_square_norm=(dFF_square-dFF_square.mean(axis=0)[None,:])/dFF_square.std(axis=0)[None,:]

swim_norm=(frame_swim_tcourse[0,:]-frame_swim_tcourse[0,:].mean())/frame_swim_tcourse[0,:].std()
corr_matrix=(dFF_square_norm*swim_norm[:,None]).sum(axis=0).reshape((dim[1],dim[2]),order='F')/dim[0]

plt.figure(1)
plt.imshow(corr_matrix)

swim_num=len(swimdata['swimStartIndT'])
thre=0.5

swim_template=np.zeros((7500,2))
swim_template_std=np.zeros((7500,2))
resp_mean=np.zeros((7500,3))
resp_std=np.zeros((7500,3))
snum=0
for i in range(2,swim_num-1):
    Sstart = swimdata['swimStartIndT'][i]
    Send = swimdata['swimEndIndT'][i]
    bdur=(Send-Sstart)
    max1=((rawdata['ch1'][Sstart:Send]).__abs__()).max()
    max2=((rawdata['ch2'][Sstart:Send]).__abs__()).max()
    interv=swimdata['swimStartIndT'][i+1]-swimdata['swimStartIndT'][i]
    
    frame_info=frame_tcourse[(Sstart-1500):(Sstart+6000)]
    frame_loc=np.where(frame_info>0)[0]
    if (frame_loc.size>0):

        frame_num=frame_info[frame_loc].astype('int')
        frame_min=frame_num.min()
        
        if ( bdur > 10 and bdur < 6000 and max(max1,max2)< thre and Sstart > 1500 and (Sstart+6000) < reclen and interv>3000 and frame_min>600 ):
            snum+=1
            swim_template[:,0]+=swimdata['fltCh1'][(Sstart-1500):(Sstart+6000)]
            swim_template[:,1]+=swimdata['fltCh2'][(Sstart-1500):(Sstart+6000)]
            swim_template_std[:,0]+=swimdata['fltCh1'][(Sstart-1500):(Sstart+6000)]**2
            swim_template_std[:,1]+=swimdata['fltCh2'][(Sstart-1500):(Sstart+6000)]**2
            
            frame_value=norm_tcourse[frame_num-1,:].T
            
            tmp1=np.interp(np.arange(7500),frame_loc,frame_value[0,:])
            tmp2=np.interp(np.arange(7500),frame_loc,frame_value[1,:])
            tmp3=np.interp(np.arange(7500),frame_loc,frame_value[2,:])
            
            resp_mean[:,0]+=tmp1-tmp1[1:1500].mean()
            resp_mean[:,1]+=tmp2-tmp2[1:1500].mean()
            resp_mean[:,2]+=tmp3-tmp3[1:1500].mean()
            
            resp_std[:,0]+=(tmp1-tmp1[1:1500].mean())**2
            resp_std[:,1]+=(tmp2-tmp2[1:1500].mean())**2
            resp_std[:,2]+=(tmp3-tmp3[1:1500].mean())**2
        
        
swim_template /= snum
swim_template_std /= snum
resp_mean /= snum
resp_std  /= snum

swim_error=np.sqrt(swim_template_std-swim_template**2)/np.sqrt(snum)
resp_error=np.sqrt(resp_std-resp_mean**2)/np.sqrt(snum)

plt.figure(2,figsize=(10,5))
ax1=plt.subplot(121)
ax1.fill_between(np.arange(-1500,6000)[::10]/6000,swim_template[::10,0]*1000-swim_error[::10,0]*1000,swim_template[::10,0]*1000+swim_error[::10,0]*1000)
plt.plot(np.arange(-1500,6000)[::10]/6000,swim_template[:,0][::10]*1000,'b')
ax1.fill_between(np.arange(-1500,6000)[::10]/6000,swim_template[::10,1]*1000-swim_error[::10,1]*1000,swim_template[::10,1]*1000+swim_error[::10,1]*1000)
plt.plot(np.arange(-1500,6000)[::10]/6000,swim_template[:,1][::10]*1000,'r')
plt.title('%s swim bouts'%(snum))
plt.ylim(0,2)


ax2=plt.subplot(122)
ax2.fill_between(np.arange(-1500,6000)[::10]/6000,resp_mean[:,0][::10]*100-resp_error[::10,0]*100,resp_mean[::10,0]*100+resp_error[::10,0]*100)
plt.plot(np.arange(-1500,6000)[::10]/6000,resp_mean[:,0][::10]*100,'c')

ax2.fill_between(np.arange(-1500,6000)[::10]/6000,resp_mean[:,1][::10]*100-resp_error[::10,1]*100,resp_mean[::10,1]*100+resp_error[::10,1]*100)
plt.plot(np.arange(-1500,6000)[::10]/6000,resp_mean[:,1][::10]*100,'m')

ax2.fill_between(np.arange(-1500,6000)[::10]/6000,resp_mean[:,2][::10]*100-resp_error[::10,2]*100,resp_mean[::10,2]*100+resp_error[::10,2]*100)
plt.plot(np.arange(-1500,6000)[::10]/6000,resp_mean[:,2][::10]*100,'y')
plt.ylim(-0.5,10)