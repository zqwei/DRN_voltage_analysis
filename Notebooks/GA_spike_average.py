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

print('Generate plot for population aligned by maximum peak')
nan_line_wid = 3
nan_line = np.empty((len(ave_low_list), nan_line_wid))
nan_line[:] = np.nan
ave_list = np.concatenate([np.array(ave_low_list), nan_line, np.array(ave_high_list)], axis=-1)
ave_list_min = np.nanmin(ave_list, axis=-1, keepdims=True)
ave_list_max = np.nanmax(ave_list, axis=-1, keepdims=True)
valid = (ave_list_max - ave_list_min) >0
ave_list = (ave_list - ave_list_min)/(ave_list_max - ave_list_min)

ave_list_ = ave_list[valid[:,0], :]
# max_ind = np.nanargmax(ave_list_, axis=-1)
# sort_max_ind = np.argsort(max_ind)
##########################
# center of mass
ave_list__ = np.concatenate([np.array(ave_low_list)[:, t_pre:], np.array(ave_high_list)[:, t_pre:]], axis=-1)[valid[:,0]]
sort_max_ind= np.argsort(np.dot(ave_list__, np.arange(ave_list__.shape[1]))/ave_list__.sum(axis=1))
plt.figure(figsize=(8, 3))
plt.imshow(ave_list_[sort_max_ind, :], aspect='auto', origin='lower')
plt.vlines([t_pre, t_pre*2+t_post+nan_line_wid], [0], [len(sort_max_ind)], linestyles='--', colors='w')
plt.ylim([0, len(sort_max_ind)])
plt.ylabel('Neuronal index')
plt.xlabel('Time from swim bout onset (sec)')
plt.xticks([t_pre, t_pre+300, t_pre*2+t_post+nan_line_wid, t_pre*2+t_post+300+nan_line_wid], ['0', '1', '0', '1'])
plt.yticks([0, len(sort_max_ind)-1], [1, len(sort_max_ind)])
plt.colorbar()
sns.despine()
# plt.savefig('../Plots/gain/pop_act_max.pdf')
plt.savefig('../Plots/gain/pop_act_center_mass.pdf')
plt.close('all')

print('Generate plot for population p-values')
# sig_mat = np.abs(p_mat)<0.05
# sel_ind = sig_mat.sum(axis=-1)>30
# p_mat = p_mat[sel_ind]
sig_mat = -np.log(np.abs(p_mat)+1e-10)
sign_mat = np.sign(p_mat)
# ind = np.argmax(sig_mat, axis=-1)
# sign_ind = np.median(sign_mat, axis=-1)
# ind = np.lexsort((ind, sign_ind))
sig_mat = sig_mat*sign_mat
# sig_mat[np.abs(sig_mat)<3] = 0
plt.figure(figsize=(8, 3))
plt.imshow(sig_mat[sort_max_ind, :], aspect='auto', origin='lower',cmap=plt.cm.RdGy_r, vmin=-10, vmax=10)
plt.vlines([t_pre], [0], [len(sort_max_ind)], linestyles='--', colors='k')
plt.ylim([0, len(sort_max_ind)])
plt.ylabel('Neuronal index')
plt.xlabel('Time from swim bout onset (sec)')
plt.xticks([t_pre, t_pre+300], ['0', '1'])
plt.yticks([0, len(sort_max_ind)-1], [1, len(sort_max_ind)])
plt.colorbar()
sns.despine()
plt.savefig('../Plots/gain/pop_act_p_value.pdf')
plt.close('all')

print('Generate plot for population p-values of selective neurons')
sig_mat = np.abs(p_mat)<0.05
## parameter for threshold
sel_ind = sig_mat.sum(axis=-1)>10
p_mat_ = np.abs(p_mat)[sel_ind]
sig_mat = -np.log(np.abs(p_mat_)+1e-10)
sign_mat = np.sign(p_mat_)
ind = np.argmax(sig_mat, axis=-1)
sign_ind = np.median(sign_mat, axis=-1)
ind = np.lexsort((ind, sign_ind))
sig_mat = sig_mat*sign_mat
# sig_mat[np.abs(sig_mat)<3] = 0
plt.figure(figsize=(8, 3))
plt.imshow(sig_mat[ind], aspect='auto', origin='lower',cmap=plt.cm.RdGy_r, vmin=-10, vmax=10)
plt.vlines([t_pre], [0], [len(ind)], linestyles='--', colors='k')
plt.ylim([0, len(ind)])
plt.ylabel('Neuronal index')
plt.xlabel('Time from swim bout onset (sec)')
plt.xticks([t_pre, t_pre+300], ['0', '1'])
plt.yticks([0, len(ind)-1], [1, len(ind)])
plt.colorbar()
sns.despine()
plt.savefig('../Plots/gain/pop_sel_act_p_value.pdf')
plt.close('all')


print('Generate plot for average spikes for selective population')
sig_mat = np.abs(p_mat)[:, t_pre-30:(t_pre+300)]<0.05
sel_ind = sig_mat.sum(axis=-1)>10
print(f'number of cells {sel_ind.sum()}')
print(f'number of fish {len(np.unique(np.array(fish_id)[sel_ind]))}')
plt.figure(figsize=(4, 3))
# low gain
ave_act = np.array(ave_low_list)[sel_ind].mean(axis=0)
sem_act = np.array(ave_low_list)[sel_ind].std(axis=0)/np.sqrt(len(ave_low_list))
plt.figure(figsize=(4, 3))
plt.plot(t_label, ave_act, '-k', lw=2)
plt.plot(t_label, ave_act+sem_act, '--k', lw=0.5)
plt.plot(t_label, ave_act-sem_act, '--k', lw=0.5)
# high gain
ave_act = np.array(ave_high_list)[sel_ind].mean(axis=0)
sem_act = np.array(ave_high_list)[sel_ind].std(axis=0)/np.sqrt(len(ave_high_list))
plt.plot(t_label, ave_act, '-r', lw=2)
plt.plot(t_label, ave_act+sem_act, '--r', lw=0.5)
plt.plot(t_label, ave_act-sem_act, '--r', lw=0.5)
# plt.vlines([0], [0], [1.1], linestyles='--', colors='k')
plt.xlim([-0.2, 1.0])
plt.ylim([0, 1.1])
plt.yticks(np.arange(0, 1.01, 0.5))
plt.xticks([0, 1])
plt.xlabel('Time from swim bout (sec)')
plt.ylabel('Spikes (/sec)')
sns.despine()
plt.savefig('../Plots/gain/pop_sel_act_ave.pdf')
plt.close('all')

print('Generate plot for average subvolts for selective population')
plt.figure(figsize=(4, 3))
# low gain
act_ = np.array(sub_low_list)
ff = np.abs(act_).max(axis=-1, keepdims=True)
act_ = act_/ff
# ave_act = act_[sel_ind].mean(axis=0)
ave_act = np.percentile(act_[sel_ind], 65, axis=0)
sem_act = sem(act_[sel_ind], axis=0)
plt.figure(figsize=(4, 3))
plt.plot(t_label, ave_act, '-k', lw=2)
plt.plot(t_label, ave_act+sem_act, '--k', lw=0.5)
plt.plot(t_label, ave_act-sem_act, '--k', lw=0.5)
# high gain
act_ = np.array(sub_high_list)
act_ = act_/ff
# ave_act = act_[sel_ind].mean(axis=0)
ave_act = np.percentile(act_[sel_ind], 65, axis=0)
sem_act = sem(act_[sel_ind], axis=0)
plt.plot(t_label, ave_act, '-r', lw=2)
plt.plot(t_label, ave_act+sem_act, '--r', lw=0.5)
plt.plot(t_label, ave_act-sem_act, '--r', lw=0.5)
plt.xlim([-0.2, 1.0])
plt.ylim([-1, 0.6])
plt.yticks(np.arange(-1, 1, 0.5))
plt.xticks([0, 1])
plt.xlabel('Time from swim bout (sec)')
plt.ylabel('Norm. dFF')
sns.despine()
plt.savefig('../Plots/gain/pop_sel_sub_ave.pdf')
plt.close('all')
