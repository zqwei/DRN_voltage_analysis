from utils import *

## motor clamp the trials only

rootDir = '/mnt/Dali/Takashi/Cezanne/SPIM_newPC/'
folder = '04182019'
fish = 'Fish1-3-delay'
dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'

# pamamters
swm_dir = dat_folder+f'{folder}/{fish}/swim/'
img_dir = rootDir+f'{folder}/{fish}/Registered'

frame_rate = 30
t_pre = 20
t_post = 30 # 1 sec according to frame-rate
t_flat = 15
t_valid = 16

# get valid swim data
frame_stimParams = np.load(swm_dir+'frame_stimParams.npy', allow_pickle=True)[()];
frame_swim_tcourse = np.load(swm_dir+"frame_swim_tcourse_series.npy", allow_pickle=True)[()];
rawdata = np.load(swm_dir+"rawdata.npy", allow_pickle=True)[()]
swimdata = np.load(swm_dir+"swimdata.npy", allow_pickle=True)[()]
reclen=len(swimdata['fltCh1'])
frame_tcourse=np.zeros((reclen,))
frame=np.where(np.diff((rawdata['ch3']>3).astype('int'))==1)[0]
for t in range(len(frame)-1):
    frame_tcourse[frame[t]:frame[t+1]]=t
swim_start = frame_tcourse[np.where(swimdata['swimStartT']>0)[0]].astype('int')
swim_end = frame_tcourse[np.where(swimdata['swimEndT']>0)[0]].astype('int')
# collect trial within t-pre, and t-post valid range
swim_end   = swim_end[((swim_start>t_pre) & (swim_start<(frame_swim_tcourse.shape[1]-t_post)))]
swim_start = swim_start[((swim_start>t_pre) & (swim_start<(frame_swim_tcourse.shape[1]-t_post)))]

# get dff data
dFF = np.load(img_dir+'/dFF_sub.npy')[()]
dim = dFF.shape
dFF = dFF.reshape((dim[0],dim[1]*dim[2]),order='F') # dFF reshaped to t x pixels

# remove pixels with low light intensity
ave = np.load(img_dir+'/stack_sub.npy')[()]
ave = ave.mean(axis=0)

plt.figure(figsize=(4, 10))
plt.imshow(ave, cmap=plt.cm.gray)
plt.axis('off')
plt.savefig('../Plots/snfr/glu_RD_sess_ave_img.pdf')
plt.close('all')

mask_ = ave>150
ave = ave.reshape((1,dim[1]*dim[2]),order='F')
include_pix=np.where(ave>150)[1]
dFF = dFF[:, include_pix]
n_pix = dFF.shape[-1]


swim_list = []
for ntype in range(3):
    _ = swim_start[np.where(frame_stimParams[2,swim_start]==ntype+1)[0]]
    valid = np.diff(_) > t_post
    swim_list.append(_[:-1][valid])

dff_list = []
pswim_list = []
vis_list = []
num_swim_list = []
t_len = int(t_pre+t_post)
for nswim in swim_list:
    num_swim = len(nswim)
    num_swim_list.append(num_swim)
    dff_ = np.zeros((num_swim, t_len, n_pix))
    pswim_ = np.zeros((num_swim, t_len))
    vis_ = np.zeros((num_swim, t_len))
    for ns, s in enumerate(nswim):
        l_swim = frame_swim_tcourse[1,(s-t_pre):(s+t_post)]
        r_swim = frame_swim_tcourse[2,(s-t_pre):(s+t_post)]
        pswim_[ns] = np.sqrt(l_swim**2 + r_swim**2)*100000 # TK: 10000
        dff_[ns] = dFF[(s-t_pre):(s+t_post), :] - dFF[(s-t_flat):s, :].mean(axis=0, keepdims=True)
        vis_[ns] = -frame_stimParams[0,(s-t_pre):(s+t_post)]*10000
    valid_ = np.ones(pswim_.shape[0]).astype('bool')
    valid_ = pswim_[:, -t_valid:].sum(axis=-1)==0
    # valid_ = valid_ & (pswim_.max(axis=-1)<2)
    valid_ = valid_ & (pswim_[:, :t_pre].sum(axis=-1)==0)
    dff_list.append(dff_[valid_])
    pswim_list.append(pswim_[valid_])
    vis_list.append(vis_[valid_])

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax = ax.flatten()
color_list = ['k', 'r', 'b']
for n in range(3):
    ax[0].plot(np.arange(-t_pre, t_post)/frame_rate, pswim_list[n].mean(axis=0), color_list[n])
    ax[1].plot(np.arange(-t_pre, t_post)/frame_rate, vis_list[n].mean(axis=0), color_list[n])
    ave_ = dff_list[n].mean(axis=0).mean(axis=-1)*100
    sem_ = sem(dff_list[n].mean(axis=0), axis=-1)*100
    ax[2].plot(np.arange(-t_pre, t_post)/frame_rate, ave_, color_list[n])
    ax[2].plot(np.arange(-t_pre, t_post)/frame_rate, ave_-sem_, color_list[n], lw=0.5)
    ax[2].plot(np.arange(-t_pre, t_post)/frame_rate, ave_+sem_, color_list[n], lw=0.5)
sns.despine()
ax[0].set_xlim([-0.2, 1.0])
ax[1].set_xlim([-0.2, 1.0])
ax[2].set_xlim([-0.2, 1.0])
ax[0].set_ylabel('Swim power')
ax[0].set_xlabel('Time from swim (sec)')
ax[1].set_ylabel('Visual input')
ax[1].set_xlabel('Time from swim (sec)')
ax[2].set_ylabel('Average dF/F')
ax[2].set_xlabel('Time from swim (sec)')
plt.savefig('../Plots/snfr/glu_RD_sess_exp.pdf')
plt.close('all')


tmp1 = []
tmp2 = []
plt.figure(figsize=(4, 3))
for n in range(3):
    ave_dff = dff_list[n].mean(axis=-1)*100
    d_ = np.argmax(ave_dff[:, t_pre+6:], axis=-1)/30 + 3/30
    if n==0:
        d_[d_>0.5]=0
    tmp2.append(d_)
plt.boxplot(tmp2)
plt.xlabel('Onset time of visual')
plt.ylabel('dF/F peak time (s)')
sns.despine()
plt.savefig('../Plots/snfr/glu_RD_visual_onset_sess_exp.pdf')
print(f_oneway(tmp2[0], tmp2[1], tmp2[2]))
plt.close('all')