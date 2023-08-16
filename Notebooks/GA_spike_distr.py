from utils import *

vol_file = '../Analysis/depreciated/analysis_sections_gain.csv'
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_xls_file = dat_xls_file.reset_index()
t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
t_label = np.arange(-t_pre, t_post)/300
t_sig = 240

k_ = gaussKernel(sigma=5)
k_sub = gaussKernel(sigma=5)
ave_low_list = []
ave_high_list = []
p_mat = []
fish_id = []
sub_low_list = []
sub_high_list = []

print('Collect data')
for ind, row in dat_xls_file.iterrows():    
    folder = row['folder']
    fish = row['fish']
    task_type = row['task']    
    _ = np.load(f'../Analysis/swim_power/{folder}_{fish}_swim_dat.npz')
    task_period = _['task_period'] 
    swim_starts = _['swim_starts'] 
    trial_valid = np.ones(len(swim_starts)).astype('bool')
    for n, n_swim in enumerate(swim_starts[:-1]):        
        # examine the swim with short inter-swim-interval
        if swim_starts[n+1] - n_swim < t_sig:    
            trial_valid[n] = False
    
    _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}_swim_voltr_dat.npz')
    # trial_valid = _['trial_valid']
    sub_swim = _['sub_swim']
    spk_swim = _['spk_swim']
    
    for n_cell in range(sub_swim.shape[0]):
        sub_list = sub_swim[n_cell]
        tmp = []
        for n_spk in sub_list:
            tmp.append(smooth(n_spk, k_sub))
        sub_list = np.array(tmp)
        # sub_list = sub_list - sub_list[:, 0:70].mean(axis=-1, keepdims=True) # (t_pre-30):t_pre
        sub_list = sub_list - sub_list[:, (t_pre-60):t_pre].mean(axis=-1, keepdims=True)
        
        spk_list = spk_swim[n_cell]
        tmp = []
        for n_spk in spk_list:
            tmp.append(smooth(n_spk, k_))
        spk_list = np.array(tmp)
        non_trial = np.isnan(spk_list).sum(axis=-1)==0
        trial_valid = trial_valid & non_trial
        # if np.isnan(spk_list[(task_period==1) & trial_valid].mean(axis=0)).sum()==0 and np.isnan(spk_list[(task_period==2) & trial_valid].mean(axis=0)).sum()==0:
        gain_stat = np.zeros(t_pre+t_post)
        for ntime in range(-t_pre, t_post):
            val, pval= ranksums(spk_list[(task_period==1) & trial_valid, t_pre+ntime], 
                                spk_list[(task_period==2) & trial_valid, t_pre+ntime])
            gain_stat[t_pre+ntime] = np.sign(-val) * pval
        p_mat.append(gain_stat)
        ave_low_list.append(spk_list[(task_period==1) & trial_valid].mean(axis=0)*300)
        ave_high_list.append(spk_list[(task_period==2) & trial_valid].mean(axis=0)*300)
        sub_low_list.append(sub_list[(task_period==1) & trial_valid].mean(axis=0)*100)
        sub_high_list.append(sub_list[(task_period==2) & trial_valid].mean(axis=0)*100)
        fish_id.append(folder+fish[:5])

# print number of fish
print(f'number of fish {len(np.unique(np.array(fish_id)))}')

ave_low_list=np.array(ave_low_list)
ave_high_list=np.array(ave_high_list)

fig, ax=plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True)
ax[0,0].hist(ave_low_list[:,t_pre-30:t_pre].mean(axis=-1))
ax[0,0].set_title('Mean before swim')
ax[0,1].hist(ave_low_list[:,t_pre+150:t_pre+200].mean(axis=-1))
ax[0,1].set_title('Mean after swim')
ax[1,0].hist(ave_high_list[:,t_pre-30:t_pre].mean(axis=-1))
ax[1,0].set_ylabel('Count')
ax[1,0].set_xlabel('Spike rate')
ax[1,1].hist(ave_high_list[:,t_pre+150:t_pre+200].mean(axis=-1))
plt.show()