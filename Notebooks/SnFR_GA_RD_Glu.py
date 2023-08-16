from utils import *

vol_file = '../Analysis/depreciated/analysis_sections_GA_RD_SnFR.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

dff_dat_folder = '../Analysis/snfr_dff_simple_center/'
frame_rate = 30
t_pre = 10 # time window pre-swim
t_post = 35 # time window post-swim
t_sig = 30 # time used for significance test after swim
t_len = t_pre+t_post
t_flat = 15
t_valid = 21
color_list = ['k', 'r', 'b']

dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'
k_sub = gaussKernel(sigma=1)
dat_list=[]
fish_list=[]

for _, row in dat_xls_file.iterrows():
    folder = row['folder']
    fish = row['fish']
    task_type = row['task']
    dff_dir = dat_folder+f'{folder}/{fish}/Data/'
    if not 'Glu' in row['area']:
        continue
    _ = np.load(f'../Analysis/swim_power/{folder}_{fish}_swim_dat.npz')
    r_swim = _['r_swim']
    l_swim = _['l_swim']
    task_period = _['task_period'].astype('int')
    task_period_ = _['swim_task_index'].astype('int')
    visu = _['visu']
    p_swim = r_swim + l_swim
    swim_starts = _['swim_starts']
    swim_ends = _['swim_ends']
    swim_len = swim_ends - swim_starts
    valid_trial = (swim_len>1) & (p_swim.sum(axis=-1)>0)
    valid_trial = valid_trial & (p_swim[:, -t_valid:].sum(axis=-1)==0)
    valid_trial = valid_trial & (p_swim[:, :t_pre].sum(axis=-1)==0)
    valid_trial = valid_trial & ~(((visu[:, :t_pre+5]<0).sum(axis=-1)>0) & (task_period_==5))
    valid_trial = valid_trial & ~(((visu[:, :t_pre+10]<0).sum(axis=-1)>0) & (task_period_==6))
    
    valid_list_=[]
    for n_ in range(1, 4):
        valid_list_.append((valid_trial & (task_period==n_)).sum())
    for n_ in range(4, 7):
        valid_list_.append((valid_trial & (task_period_==n_)).sum())
    if (np.array(valid_list_)==0).sum()>1:
        continue
    
    ## GA filter
    gain_stat_len = 10
    gain_sig_stat = np.ones(gain_stat_len)
    for ntime in range(gain_stat_len):
        val, pval= ranksums(p_swim[valid_trial & (task_period==1), t_pre+ntime], p_swim[valid_trial & (task_period==3), t_pre+ntime])
        gain_sig_stat[ntime] = (pval<0.05)
    if gain_sig_stat.mean()<0.3:
        continue
    
    ## RA filter
    gain_stat_len = 10
    gain_sig_stat = np.ones(gain_stat_len)
    for ntime in range(gain_stat_len):
        val, pval= ranksums(p_swim[valid_trial & (task_period_==4), t_pre+ntime], p_swim[valid_trial & (task_period_==6), t_pre+ntime])
        gain_sig_stat[ntime] = (val>0) and (pval<0.05)
    if gain_sig_stat.mean()>0:
        continue

    if not os.path.exists(dff_dat_folder+f'{folder}_{fish}_snfr_dff_dat.npz'):
        continue

    _ = np.load(dff_dat_folder+f'{folder}_{fish}_snfr_dff_dat.npz', allow_pickle=True)
    ave = _['Y_mean']
    dFF_ = _['dFF_ave'][:, np.newaxis]
    n_pix=dFF_.shape[-1]
    num_swim = len(swim_starts)
    num_comp = dFF_.shape[-1]
    for n_c in range(num_comp):
        dFF_[:, n_c]=smooth(dFF_[:, n_c], k_sub)
    

    for n_c in range(num_comp):
        c_dat=[]
        dff_ = np.zeros((num_swim, t_len))
        for ns, s in enumerate(swim_starts):
            dff_[ns] = dFF_[(s-t_pre):(s+t_post), n_c] - dFF_[(s-t_flat):s, n_c].mean(axis=0, keepdims=True)

        gain_stat_len = 10
        gain_sig_stat = np.ones(gain_stat_len)
        for ntime in range(gain_stat_len):
            val, pval= ranksums(dff_[valid_trial & (task_period==1), t_pre+ntime], dff_[valid_trial & (task_period==3), t_pre+ntime])
            gain_sig_stat[ntime] = (pval<0.05)  # (val>0) and 
        if gain_sig_stat.mean()<=0:
            continue


        color_=['k', 'r', 'r']
        plt.figure(figsize=(4, 3))
        for n in [1, 3]:
            idx = valid_trial & (task_period==n)
            mean_=np.mean(dff_[idx], axis=0)
            sem_ = sem(dff_[idx], axis=0)
            shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color=color_[n-1])
        plt.title('Gain adaptation')
        plt.xlim([-0.1, 1.0])
        # plt.ylim([-0.3, 1.0])
        sns.despine()
        plt.xlabel('Time from swim (s)')
        plt.ylabel('Glu release Norm. $\Delta$F/F')
        plt.savefig(f'../Plots/snfr/GA_RG/glu_GA_RD-GA_{folder}_{fish}.pdf')
        plt.close('all')
        
        color_=['k', 'r', 'b']
        plt.figure(figsize=(4, 3))
        for n in [4, 5, 6]:
            idx = valid_trial & (task_period_==n)
            mean_=np.mean(dff_[idx], axis=0)
            sem_ = sem(dff_[idx], axis=0)
            shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color=color_[n-4])
        plt.title('Gain adaptation')
        plt.xlim([-0.1, 1.0])
        # plt.ylim([-0.3, 1.0])
        sns.despine()
        plt.xlabel('Time from swim (s)')
        plt.ylabel('Glu release Norm. $\Delta$F/F')
        plt.savefig(f'../Plots/snfr/GA_RG/glu_GA_RD-RD_{folder}_{fish}.pdf')
        plt.close('all')
        
#         for n in [1, 3]:
#             idx = valid_trial & (task_period==n)
#             c_dat.append(np.mean(dff_[idx], axis=0).T) #-np.mean(dff_[idx, (t_pre-5):t_pre])
#         for n in [4, 5, 6]:
#             idx = valid_trial & (task_period_==n)
#             c_dat.append(np.mean(dff_[idx], axis=0).T)
#         dat_list.append(np.array(c_dat))
#         fish_list.append(folder+fish) #[:5]

# print(np.unique(fish_list).T)

# dat_list=np.array(dat_list)
# ff = dat_list[:,0].max(axis=-1, keepdims=True)
# dat_list = dat_list/ff[:,None,:]
# ff_idx = np.argmax(dat_list[:,0],axis=-1)
# idx=(ff_idx<t_pre+20) & (dat_list[:,0].min(axis=-1)>-0.8)

# plt.figure(figsize=(4, 3))
# mean_ = np.mean(dat_list[idx,0], axis=0)
# sem_ = sem(dat_list[idx,0])
# shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color='k')
# mean_ = np.mean(dat_list[idx,1], axis=0)
# sem_ = sem(dat_list[idx,1])
# shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color='r')
# plt.title('Gain adaptation')
# plt.xlim([-0.1, 1.0])
# plt.ylim([-0.3, 1.0])
# sns.despine()
# plt.xlabel('Time from swim (s)')
# plt.ylabel('Glu release Norm. $\Delta$F/F')
# plt.savefig('../Plots/snfr/GA_RG/glu_GA_RD-GA.pdf')
# plt.close('all')

# plt.figure(figsize=(4, 3))
# idx=(ff_idx<t_pre+20)
# mean_ = np.mean(dat_list[idx,2], axis=0)
# sem_ = sem(dat_list[idx,2])
# shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color='k')
# mean_ = np.mean(dat_list[idx,3], axis=0)
# sem_ = sem(dat_list[idx,3])
# shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color='r')
# mean_ = np.mean(dat_list[idx,4], axis=0)
# sem_ = sem(dat_list[idx,4])
# shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color='b')
# plt.title('Random gain')
# plt.xlim([-0.1, 1.0])
# plt.ylim([-0.3, 1.0])
# sns.despine()
# plt.xlabel('Time from swim (s)')
# plt.ylabel('Glu release Norm. $\Delta$F/F')
# plt.savefig('../Plots/snfr/GA_RG/glu_GA_RD-RD.pdf')
# plt.close('all')
