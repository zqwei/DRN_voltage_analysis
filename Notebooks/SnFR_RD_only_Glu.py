from utils import *

vol_file = '../Analysis/depreciated/analysis_sections_RD_only_SnFR.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')

dff_dat_folder = '../Analysis/snfr_dff_simple/'
frame_rate = 30
t_pre = 10 # time window pre-swim
t_post = 35 # time window post-swim
t_sig = 30 # time used for significance test after swim
t_len = t_pre+t_post
t_flat = 15
t_valid = 21
color_list = ['k', 'r', 'b']

dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'
valid_sess_list=['04182019Fish1-3-delayGlu', '05012019Fish2-2-RandomDelayGlu', '05012019Fish2-4-RandomDelayGlu']
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
    valid_trial = valid_trial & ~(((visu[:, :t_pre]<0).sum(axis=-1)>0))
    valid_trial = valid_trial & ~(((visu[:, :t_pre+5]<0).sum(axis=-1)>0) & (task_period_==2))
    valid_trial = valid_trial & ~(((visu[:, :t_pre+10]<0).sum(axis=-1)>0) & (task_period_==3))
    if not (folder + fish + row['area']) in valid_sess_list:
        continue
    
    ## RA filter
    gain_stat_len = 10
    gain_sig_stat = np.ones(gain_stat_len)
    for ntime in range(gain_stat_len):
        val, pval= ranksums(p_swim[valid_trial & (task_period_==1), t_pre+ntime], p_swim[valid_trial & (task_period_==3), t_pre+ntime])
        gain_sig_stat[ntime] = (val>0) and (pval<0.05)
    if gain_sig_stat.mean()>0:
        continue
        
    _ = np.load(dff_dat_folder+f'{folder}_{fish}_snfr_dff_dat.npz', allow_pickle=True)
    ave = _['Y_mean']
    dFF_ = _['dFF']# _['dFF_ave'][:, np.newaxis] ##
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
        for n in range(1,4):
            idx = valid_trial & (task_period_==n)
            c_dat.append(np.mean(dff_[idx], axis=0).T)
        dat_list.append(np.array(c_dat))
        fish_list.append(folder+fish[:5])

dat_list=np.array(dat_list)
ff = dat_list[:,0].max(axis=-1, keepdims=True)
dat_list = dat_list/ff[:,None,:]
# ff_idx = np.argmax(dat_list[:,0],axis=-1)
# idx=(ff_idx<t_pre+20) & (dat_list[:,0].min(axis=-1)>-0.8)
idx=(dat_list[:,0].min(axis=-1)>-1.8)

plt.figure(figsize=(4, 3))
for n_ in range(3):
    mean_ = np.mean(dat_list[idx,n_], axis=0)
    sem_ = sem(dat_list[idx,n_])
    shaded_errorbar(np.arange(-t_pre, t_post)/frame_rate, mean_, sem_, ax=plt, color=color_list[n_])

plt.title('Random delay')
plt.xlim([-0.1, 1.0])
# plt.ylim([-0.3, 1.0])
sns.despine()
plt.xlabel('Time from swim (s)')
plt.ylabel('Glu release Norm. $\Delta$F/F')
plt.savefig('../Plots/snfr/GA_RG/glu_RD_only.pdf')
plt.close('all')