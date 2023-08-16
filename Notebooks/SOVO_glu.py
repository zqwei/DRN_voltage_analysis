from utils import *

vol_file = '../SnFR_data/SnFR_Log_DRN_Exp.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'

frame_rate = 30
t_pre = 20
t_post = 30 # 1 sec according to frame-rate
t_flat = 15
t_valid = 16
color_list = ['k', 'r', 'b']

k_sub = gaussKernel(sigma=0.3)

dff_dat_folder = '../Analysis/snfr_dff_simple_center/'
t_pre = 10 # time window pre-swim
t_post = 35 # time window post-swim
t_label = np.arange(-t_pre, t_post)/30
c_list = ['k', 'r', 'b']
labels = ['CL', 'Swim-only', 'Visual-only']
t_swim_CL = t_pre + 10
t_swim_OL = t_pre + 20
t_len = t_pre+t_post

dff_list=[]
fish_list=[]

for ind, row in dat_xls_file.iterrows():
    if row['task'] != 'Swimonly_Visualonly':
        continue
    if row['area'] != 'Glu':
        continue
    folder = row['folder']
    fish = row['fish']
    task_type = row['task']
    rootDir = row['rootDir']
    img_dir = rootDir+f'{folder}/{fish}/Registered'
    dff_dir = dat_folder+f'{folder}/{fish}/Data/'
    if not os.path.exists(f'../Analysis/swim_power/{folder}_{fish}_swim_dat.npz'):
        continue
    _ = np.load(f'../Analysis/swim_power/{folder}_{fish}_swim_dat.npz')
    r_swim = _['r_swim']
    l_swim = _['l_swim']
    swim_starts = _['swim_starts']
    task_period = _['swim_task_index'].astype('int')
    visu = _['visu']
    p_swim = np.sqrt(r_swim**2 + l_swim**2)    
    task_period = _['swim_task_index'].astype('int')            
    # _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}_swim_voltr_dat.npz')
    trial_valid = np.ones(len(p_swim)).astype('bool')
    swim_power_thres = np.percentile(p_swim[(task_period==1) & trial_valid].mean(axis=0), 99)
    
    trial_pre = (p_swim[:, :t_pre]>0).sum(axis=-1)==0
    trial_valid_CL = (p_swim[:, t_swim_CL:t_swim_CL+15]>0).sum(axis=-1)==0
    trial_valid_CL = trial_valid_CL & trial_pre
    trial_valid_OL = ((visu.max(axis=-1, keepdims=True)-visu)[:, :-5]>0).sum(axis=-1)==0
    trial_valid_OL = trial_valid_OL & (p_swim[:, t_swim_CL:t_pre+30].max(axis=-1)<swim_power_thres)
    trial_valid_OL = trial_valid_OL & trial_pre
    trial_valid_OL = trial_valid_OL & ((p_swim[:, t_swim_OL:t_pre+30]>1).sum(axis=-1)==0)
    trial_valid_OL = trial_valid_OL & ((p_swim[:, t_swim_OL:t_swim_OL+15]>0).sum(axis=-1)==0)
    trial_valid_VL = (p_swim[:, t_pre:t_pre+30]>0).sum(axis=-1)==0
    trial_valid_VL = trial_valid_VL & (visu[:, t_swim_OL:t_pre+30].min(axis=-1)>=0)
    trial_valid_VL = trial_valid_VL & (visu[:, t_pre:t_swim_OL].min(axis=-1)<0)
    trial_valid_VL = trial_valid_VL & trial_pre
    
    for n in range(3):
        if n==0:
            trial_valid_ = trial_valid & trial_valid_CL
        if n==1:
            trial_valid_ = trial_valid & trial_valid_OL
        if n==2:
            trial_valid_ = trial_valid & trial_valid_VL
        if ((task_period==n+1) & trial_valid_).sum()<15:
            continue
    if ((task_period==n+1) & trial_valid_).sum()<15:
        continue
    
    
    if not os.path.exists(dff_dat_folder+f'{folder}_{fish}_snfr_dff_dat.npz'):
        continue
    
    _ = np.load(dff_dat_folder+f'{folder}_{fish}_snfr_dff_dat.npz', allow_pickle=True)
    ave = _['Y_mean']
    dFF_ = _['dFF_ave'][:, np.newaxis]
    n_pix=dFF_.shape[-1]
    
#     _ = np.load(dff_dir+'components.npz');
#     A_ = _['A_']
#     C_ = _['C_']
#     dFF_ = (C_/np.median(C_, axis=-1, keepdims=True)-1).T
#     n_pix = dFF_.shape[-1]
    
    for dFF in dFF_.T:
        c_dff=[]
        # dFF = smooth(dFF, k_sub)
        dff_ = np.zeros((len(swim_starts), t_len))
        for ns, s in enumerate(swim_starts):
            dff_[ns] = dFF[(s-t_pre):(s+t_post)] - dFF[(s-t_flat):s].mean(axis=0, keepdims=True)

        if np.percentile(p_swim[(task_period==2) & trial_valid].mean(axis=0), 95)>swim_power_thres:
            continue

        if ((task_period==2) & trial_valid & trial_valid_OL).sum()<5:
            continue
            
        color_=['k', 'r', 'b']
        plt.figure(figsize=(4, 3))
        
        for n in range(3):
            if n==0:
                trial_valid_ = trial_valid & trial_valid_CL
            if n==1:
                trial_valid_ = trial_valid & trial_valid_OL
            if n==2:
                trial_valid_ = trial_valid & trial_valid_VL
            mean_=np.mean(dff_[(task_period==n+1) & trial_valid_], axis=0)
            sem_ = sem(dff_[(task_period==n+1) & trial_valid_], axis=0)
            shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color=color_[n-1])
#             c_dff.append(np.mean(dff_[(task_period==n+1) & trial_valid_], axis=0))
#         dff_list.append(np.array(c_dff))
#         fish_list.append(folder+fish) #[:5]
        plt.xlim([-0.1, 1.0])
        # plt.ylim([-0.3, 1.0])
        sns.despine()
        plt.xlabel('Time from swim (s)')
        plt.ylabel('Glu $\Delta$F/F')
        plt.savefig(f'../Plots/sovo/glu_{folder}_{fish}.pdf')
        plt.close('all')

# print(np.unique(fish_list).T)

# dat_list=np.array(dff_list)*100
# ff = dat_list[:,0].max(axis=-1, keepdims=True)
# dat_list = dat_list/ff[:,None,:]

# plt.figure(figsize=(4, 3))
# mean_ = np.mean(dat_list[:,0], axis=0)
# sem_ = sem(dat_list[:,0])
# shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color='k')
# mean_ = np.mean(dat_list[:,1], axis=0)
# sem_ = sem(dat_list[:,1])
# shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color='r')
# mean_ = np.mean(dat_list[:,2], axis=0)
# sem_ = sem(dat_list[:,1])
# shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color='b')
# plt.title('SOVO')
# plt.xlim([-0.1, 1.0])
# plt.ylim([-0.3, 1.0])
# sns.despine()
# plt.xlabel('Time from swim (s)')
# plt.ylabel('GABA release Norm. $\Delta$F/F')
# plt.savefig('../Plots/sovo/glu.pdf')

# print(np.unique(fish_list))
# print(dat_list.shape)