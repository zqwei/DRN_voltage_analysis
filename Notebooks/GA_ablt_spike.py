from utils import *

vol_file = '../Analysis/depreciated/analysis_sections_ablation_gain.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
t_len = t_pre+t_post
t_sig = 300 # time used for significance test after swim
non_spike_thres = 100


def dynamics_ablt(control_=False, c_str=''):
    ## find selective neurons
    valid_ind = np.zeros(len(dat_xls_file)).astype('bool')
    k_ = gaussKernel(sigma=7)
    gain_stat_len = 300
    p_mat_before_ = []
    p_mat_after_ = []
    p_mat_short = []

    for ind, row in dat_xls_file.iterrows():
        ablation_pair = search_paired_data(row, dat_xls_file)
        if not ablation_pair:
            continue
        folder = row['folder']
        fish = row['fish'][:-6]
        task_type = row['task']
        if ('control' in task_type)==control_:
            continue 
        ##### valid cell
        try:
            _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}_swim_voltr_valid_cell.npz')
            valid_cell = _['valid_cell']
        except:
            continue
        ##### before
        _ = np.load(f'../Analysis/swim_power/{folder}_{fish}before_swim_dat.npz')
        task_period = _['task_period']     
        _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}before_swim_voltr_ablt_dat.npz')
        sub_swim = _['sub_swim']
        spk_swim = _['spk_swim']
        sub_sig_swim = _['sub_sig_swim']
        trial_valid = _['trial_valid']

        ##### after
        _ = np.load(f'../Analysis/swim_power/{folder}_{fish}after_swim_dat.npz')
        task_period_after = _['task_period']
        _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}after_swim_voltr_ablt_dat.npz')
        sub_swim_after = _['sub_swim']
        spk_swim_after = _['spk_swim']
        sub_sig_swim_after = _['sub_sig_swim']
        trial_valid_after = _['trial_valid']

        if trial_valid_after.sum()<30:
            continue

        ### plot combined

        for n_cell in range(sub_swim.shape[0]):
            if not valid_cell[n_cell]:
                continue

            spk_list = spk_swim[n_cell]
            tmp = []
            for n_spk in spk_list:
                tmp.append(smooth(n_spk, k_))
            spk_list = np.array(tmp)
            gain_stat = np.zeros(t_len)
            gain_stat[:] = np.nan
            if (task_period==2).sum()>0:
                for ntime in range(-t_pre, t_post):
                    val, pval= ranksums(spk_list[(task_period==1) & trial_valid, t_pre+ntime], 
                                        spk_list[(task_period==2) & trial_valid, t_pre+ntime])
                    gain_stat[ntime] = np.sign(-val) * pval
                v_s, gain_stat_short = ranksums(spk_list[(task_period==1) & trial_valid, t_pre:], 
                                        spk_list[(task_period==2) & trial_valid, t_pre:])
                gain_stat_short = np.sign(-v_s) * gain_stat_short
                p_mat_before_.append(gain_stat)
                tmp_ = [gain_stat_short]

            spk_list = spk_swim_after[n_cell]
            tmp = []
            for n_spk in spk_list:
                tmp.append(smooth(n_spk, k_))
            spk_list = np.array(tmp)
            gain_stat = np.zeros(gain_stat_len)
            if (task_period==2).sum()>0:
                for ntime in range(gain_stat_len):
                    val, pval= ranksums(spk_list[(task_period_after==1) & trial_valid_after, t_pre+ntime], 
                                        spk_list[(task_period_after==2) & trial_valid_after, t_pre+ntime])
                    gain_stat[ntime] = np.sign(val) * pval
                v_s, gain_stat_short = ranksums(spk_list[(task_period_after==1) & trial_valid_after, t_pre:].mean(axis=-1), 
                                        spk_list[(task_period_after==2) & trial_valid_after, t_pre:].mean(axis=-1))
                gain_stat_short = np.sign(-v_s) * gain_stat_short
                p_mat_after_.append(gain_stat)
                tmp_.append(gain_stat_short)       
                p_mat_short.append(tmp_)
    
    sig_thres = 10
    p_mat = np.array(p_mat_before_)
    sig_mat = np.abs(p_mat)<0.05
    sel_ind = sig_mat.sum(axis=-1)>sig_thres
    p_mat = np.array(p_mat_after_)
    sig_mat = np.abs(p_mat)<0.05
    sel_ind_ = sig_mat.sum(axis=-1)>sig_thres
    
    k_ = gaussKernel(sigma=1)
    valid_ind = np.zeros(len(dat_xls_file)).astype('bool')
    gain_list = []
    cell_before = []
    cell_after = []
    diff_act = []
    fish_list = []
    ### spike plots
    for ind, row in dat_xls_file.iterrows():
        ablation_pair = search_paired_data(row, dat_xls_file)
        if not ablation_pair:
            continue
        folder = row['folder']
        fish = row['fish'][:-6]
        task_type = row['task']
        if ('control' in task_type)==control_:
            continue     
        try:
            _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}_swim_voltr_valid_cell.npz')
            valid_cell = _['valid_cell']
        except:
            continue
        ##### before
        _ = np.load(f'../Analysis/swim_power/{folder}_{fish}before_swim_dat.npz')
        task_period = _['task_period']            
        _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}before_swim_voltr_ablt_dat.npz')
        sub_swim = _['sub_swim']
        spk_swim = _['spk_swim']
        sub_sig_swim = _['sub_sig_swim']
        trial_valid = _['trial_valid']
        ##### after
        _ = np.load(f'../Analysis/swim_power/{folder}_{fish}after_swim_dat.npz')
        task_period_after = _['task_period']
        _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}after_swim_voltr_ablt_dat.npz')
        sub_swim_after = _['sub_swim']
        spk_swim_after = _['spk_swim']
        sub_sig_swim_after = _['sub_sig_swim']
        trial_valid_after = _['trial_valid']

        if trial_valid_after.sum()<30:
            continue
        for n_cell in range(sub_swim.shape[0]): 
            if not valid_cell[n_cell]:
                continue
            fish_list.append(folder+fish[:5])
            spk_list = spk_swim[n_cell]
            tmp = []
            for n_spk in spk_list:
                tmp.append(smooth(n_spk, k_))
            spk_list = np.array(tmp)

            ave_ = spk_list[trial_valid, :]*300
            ave1 = spk_list[(task_period==1) & trial_valid, :]*300
            ave2 = spk_list[(task_period==2) & trial_valid, :]*300
            mean_ = np.nanmean(ave_, axis=0)
            cell_before.append([np.nanmean(ave_, axis=0), np.nanmean(ave1, axis=0), np.nanmean(ave2, axis=0)])
            diff_act.append((np.nanmean(ave2, axis=0)-np.nanmean(ave1, axis=0)).sum())
            spk_list = spk_swim_after[n_cell]
            tmp = []
            for n_spk in spk_list:
                tmp.append(smooth(n_spk, k_))
            spk_list = np.array(tmp)
            ave_ = spk_list[trial_valid_after, :]*300
            ave1 = spk_list[(task_period_after==1) & trial_valid_after, :]*300
            ave2 = spk_list[(task_period_after==2) & trial_valid_after, :]*300
            mean_ = np.nanmean(ave_, axis=0)
            cell_after.append([np.nanmean(ave_, axis=0), np.nanmean(ave1, axis=0), np.nanmean(ave2, axis=0)])
    
    sel_all = sel_ind | sel_ind_
    fish_list = np.array(fish_list)
    fishid = np.unique(fish_list)
    plt.figure(figsize=(4, 3))
    tmp_ind = sel_all & (np.array(diff_act)>0)
    mean_ = np.mean(np.array(cell_before)[tmp_ind, 1], axis=0)
    std_ = sem(np.array(cell_before)[tmp_ind, 1], axis=0)
    shaded_errorbar(np.arange(-t_pre, t_post)/300, mean_, std_, color='k')
    mean_ = np.mean(np.array(cell_before)[tmp_ind, 2], axis=0)
    std_ = sem(np.array(cell_before)[tmp_ind, 2], axis=0)
    shaded_errorbar(np.arange(-t_pre, t_post)/300, mean_, std_, color='r')
    sns.despine()
    plt.ylabel('Spikes (/sec)')
    plt.xlabel('Time from swim onset (sec)')
    plt.xlim([-0.2, 1.0])
    plt.ylim([0, 1.1])
    plt.savefig(f'../Plots/gain_ablt/pop_ave_before_ablt{c_str}.pdf')
    plt.figure(figsize=(4, 3))
    mean_ = np.mean(np.array(cell_after)[tmp_ind, 1], axis=0)
    std_ = sem(np.array(cell_after)[tmp_ind, 1], axis=0)
    shaded_errorbar(np.arange(-t_pre, t_post)/300, mean_, std_, color='darkgray')
    mean_ = np.mean(np.array(cell_after)[tmp_ind, 2], axis=0)
    std_ = sem(np.array(cell_after)[tmp_ind, 2], axis=0)
    shaded_errorbar(np.arange(-t_pre, t_post)/300, mean_, std_, color='salmon')
    sns.despine()
    plt.ylabel('Spikes (/sec)')
    plt.xlabel('Time from swim onset (sec)')
    plt.xlim([-0.2, 1.0])
    plt.ylim([0, 1.1])
    plt.savefig(f'../Plots/gain_ablt/pop_ave_after_ablt{c_str}.pdf')
    
    print(f'number of cells: {sum(tmp_ind)}')
    print(f'number of fish: {np.unique(np.array(fish_list)[tmp_ind]).shape[0]}')
    
    ### subvolt plots
    k_sub = gaussKernel(sigma=5)
    valid_ind = np.zeros(len(dat_xls_file)).astype('bool')
    cell_before = []
    cell_after = []
    t_label = np.arange(-t_pre, t_post)/300

    for ind, row in dat_xls_file.iterrows():
        ablation_pair = search_paired_data(row, dat_xls_file)
        if not ablation_pair:
            continue
        folder = row['folder']
        fish = row['fish'][:-6]
        task_type = row['task']
        if ('control' in task_type)==control_:
            continue 
        ##### valid cell
        try:
            _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}_swim_voltr_valid_cell.npz')
            valid_cell = _['valid_cell']
        except:
            continue
        ##### before
        _ = np.load(f'../Analysis/swim_power/{folder}_{fish}before_swim_dat.npz')
        task_period = _['task_period']  
        swim_starts = _['swim_starts'] 
        trial_valid_ = np.ones(len(swim_starts)).astype('bool')
        for n, n_swim in enumerate(swim_starts[:-1]):        
            if swim_starts[n+1] - n_swim < t_sig:    
                trial_valid_[n] = False
        trial_valid = trial_valid_.copy()
        _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}before_swim_voltr_ablt_dat.npz')
        sub_swim = _['sub_swim']
        spk_swim = _['spk_swim']
        sub_sig_swim = _['sub_sig_swim']
        ##### after
        _ = np.load(f'../Analysis/swim_power/{folder}_{fish}after_swim_dat.npz')
        task_period_after = _['task_period']
        swim_starts = _['swim_starts'] 
        trial_valid_ = np.ones(len(swim_starts)).astype('bool')
        for n, n_swim in enumerate(swim_starts[:-1]):        
            if swim_starts[n+1] - n_swim < t_sig:    
                trial_valid_[n] = False
        trial_valid_after = trial_valid_.copy()
        _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}after_swim_voltr_ablt_dat.npz')
        sub_swim_after = _['sub_swim']
        spk_swim_after = _['spk_swim']
        sub_sig_swim_after = _['sub_sig_swim']
        if trial_valid_after.sum()<30:
            continue
        for n_cell in range(sub_swim.shape[0]):
            if not valid_cell[n_cell]:
                continue

            sub_sig = sub_sig_swim[n_cell]
            sub_list = sub_swim[n_cell]
            tmp = []
            for n_spk in sub_list:
                tmp.append(smooth(n_spk, k_sub))
            sub_list = np.array(tmp)
            sub_list = sub_list - sub_list[:, (t_pre-60):(t_pre-30)].mean(axis=-1, keepdims=True)
            spk_list = spk_swim[n_cell]
            ave_ = sub_list[(task_period==1) & trial_valid, :]*100
            mean_ = np.nanmedian(ave_, axis=0)
            cell_before.append(mean_)
            sub_sig = sub_sig_swim_after[n_cell]
            sub_list = sub_swim_after[n_cell]
            sub_list = sub_list - sub_list[:, 0:70].mean(axis=-1, keepdims=True)
            spk_list = spk_swim_after[n_cell]
            ave_ = sub_list[(task_period_after==1) & trial_valid_after, :]*100
            mean_pre = mean_
            mean_ = np.nanmedian(ave_, axis=0)
            ff = np.abs(mean_).max()/np.abs(mean_pre).max()
            mean_ = mean_/ff
            cell_after.append(mean_)
    
    plt.figure(figsize=(4, 3))
    act_ = np.array(cell_before)
    ff = np.abs(act_).max(axis=-1, keepdims=True)
    act_ = act_ / ff
    t_pad=30
    act_valid = (act_[:,t_pre:(t_pre+t_pad)]<0).sum(axis=-1)==t_pad
    mean_ = np.mean(act_[tmp_ind & act_valid], axis=0)
    offset_ = mean_[t_pre-30]
    std_ = sem(act_[tmp_ind], axis=0)
    shaded_errorbar(np.arange(-t_pre, t_post)/300, mean_, std_, color='b')
    act_ = np.array(cell_after)
    act_ = act_ / ff
    mean_ = np.mean(act_[tmp_ind & act_valid], axis=0)
    mean_ = mean_ - (mean_[t_pre-30]-offset_)
    std_ = sem(act_, axis=0)
    shaded_errorbar(np.arange(-t_pre, t_post)/300, mean_, std_, color='g')
    sns.despine()
    plt.ylabel('Normalized dF/F')
    plt.xlabel('Time from swim onset (sec)')
    plt.xlim([-0.2, 1.0])
    plt.yticks(np.arange(-0.6, 0.6, 0.2))
    plt.savefig(f'../Plots/gain_ablt/pop_ave_sub_ablt{c_str}.pdf')


# dynamics_ablt(control_=False, c_str='_control')
dynamics_ablt(control_=True, c_str='')