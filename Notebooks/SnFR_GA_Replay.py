from utils import *

vol_file = '../Analysis/depreciated/analysis_sections_GA_Replay_SnFR.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

dff_dat_folder = '../Analysis/snfr_dff_simple_center/'
frame_rate = 30
t_pre = 10 # time window pre-swim
t_post = 35 # time window post-swim
t_sig = 30 # time used for significance test after swim
t_len = t_pre+t_post
t_flat = 5
t_valid = 21
color_list = ['k', 'r', 'b']

k_sub = gaussKernel(sigma=0.3)
dat_list=[]
fish_list=[]

for _, row in dat_xls_file.iterrows():
    folder = row['folder']
    fish = row['fish']
    task_type = row['task']
    _ = np.load(f'../Analysis/swim_power/{folder}_{fish}_swim_dat.npz')
    r_swim = _['r_swim']
    l_swim = _['l_swim']
    task_period = _['task_period'].astype('int')
    task_period_ = _['swim_task_index'].astype('int')
    visu = _['visu']
    p_swim = r_swim + l_swim
    swim_starts = _['swim_starts']
    swim_ends = _['swim_ends']
    dist_=_['dist_']
    swim_len = swim_ends - swim_starts
    valid_trial = (swim_len>1) & (((p_swim.sum(axis=-1)>0) & (task_period!=6)) | (task_period==6))
    valid_trial = valid_trial & (p_swim[:, -t_valid:].sum(axis=-1)==0)
    valid_trial = valid_trial & (p_swim[:, :t_pre].sum(axis=-1)==0)
    valid_trial = valid_trial & ~((task_period==6) & (p_swim.sum(axis=-1)>0))
    
    ## GA filter
    gain_stat_len = 10
    gain_sig_stat = np.ones(gain_stat_len)
    for ntime in range(gain_stat_len):
        val, pval= ranksums(p_swim[valid_trial & (task_period==1), t_pre+ntime], p_swim[valid_trial & (task_period==2), t_pre+ntime])
        gain_sig_stat[ntime] = (pval<0.05)
    if gain_sig_stat.mean()<0.3:
        continue
        
    starts_before = swim_starts[valid_trial & (task_period==2)]
    starts_after = swim_starts[valid_trial & (task_period==6)]
    idx = (np.where((starts_after[:, None]-starts_before.T)==dist_))
    starts_before=starts_before[idx[1]]
    starts_after=starts_after[idx[0]]
    idx_before=np.where(valid_trial & (task_period==2))[0][idx[1]]
    idx_after=np.where(valid_trial & (task_period==6))[0][idx[0]]

        
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
            val, pval= ranksums(dff_[valid_trial & (task_period==1), t_pre+ntime], dff_[valid_trial & (task_period==2), t_pre+ntime])
            gain_sig_stat[ntime] = (val>0) and (pval<0.05)  # 

        if gain_sig_stat.mean()<0.2:
            continue
        
        plt.figure(figsize=(4, 3))
        for n_, n in enumerate([1, 2, 6]):
            idx = valid_trial & (task_period==n)
            mean_=np.mean(dff_[idx], axis=0)
            sem_ = sem(dff_[idx], axis=0)
            shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color=color_list[n_])
        plt.title('Gain adaptation')
        plt.xlim([-0.1, 1.0])
        # plt.ylim([-0.3, 1.0])
        sns.despine()
        plt.xlabel('Time from swim (s)')
        plt.ylabel('Glu release Norm. $\Delta$F/F')
        plt.savefig(f'../Plots/snfr/GA_RG/glu-GA-Replay_{folder}_{fish}.pdf')
        plt.close('all')
            