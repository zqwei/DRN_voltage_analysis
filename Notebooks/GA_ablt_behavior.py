from utils import *

vol_file = '../Analysis/depreciated/analysis_sections_gain.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
gain_stat_len = 100
sig_thres = 0.1
vol_file = '../Analysis/depreciated/analysis_sections_ablation_gain.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
t_pre = 100 # time window pre-swim
t_post = 550 # time window post-swim
t_sig = 300 # time used for significance test after swim


def behave_compare(control_=False, c_str=''):
    swim_len_list = []
    swim_type_list = []
    swim_power_list = []
    fish_id = []
    swim_sig = []
    ablt_id = []

    for _, row in dat_xls_file.iterrows():
        folder = row['folder']
        fish = row['fish']
        task_type = row['task']
        _ = np.load(f'../Analysis/swim_power/{folder}_{fish}_swim_dat.npz')
        r_swim = _['r_swim']
        l_swim = _['l_swim']
        task_period = _['task_period'].astype('int')
        if ('control' in task_type)==control_:
            continue
        visu = _['visu']
        p_swim = r_swim + l_swim
        swim_starts = _['swim_starts']
        swim_ends = _['swim_ends']
        swim_len = swim_ends - swim_starts
        gain_stat = np.zeros(gain_stat_len)
        gain_sig_stat = np.ones(gain_stat_len)
        if (task_period==2).sum()>0:
            for ntime in range(gain_stat_len):
                val, pval= ranksums(p_swim[task_period==1, t_pre+ntime], p_swim[task_period==2, t_pre+ntime])
                gain_stat[ntime] = np.sign(val) * pval
                gain_sig_stat[ntime] = (val>0) and (pval<0.05)
        if 'before' in task_type:
            tt = 'before'
        if 'after' in task_type:
            tt = 'after'
        swim_power = np.zeros(len(swim_len))
        swim_power[:] = np.nan
        for n_swim in range(len(swim_len)):
            if swim_len[n_swim]>0:
                swim_power[n_swim] = p_swim[n_swim, t_pre:(t_pre+swim_len[n_swim])].mean()
        if (tt=='before') and (gain_sig_stat.mean()<sig_thres):
            continue  
        if (tt=='after') and (not (folder[:4]+'_'+fish[:5]) in fish_id):
            continue
        swim_type_list.extend(task_period)
        swim_sig.extend([gain_sig_stat.mean()]*len(swim_len))
        fish_id.extend([folder[:4]+'_'+fish[:5]]*len(swim_len))
        ablt_id.extend([tt]*len(swim_len))
        # fish_id.extend([folder+'_'+fish]*len(swim_len))
        swim_len_list.extend(swim_len)
        swim_power_list.extend(swim_power)
    fish_behavior = {'fish_id':fish_id, 
                     'ablt_id':ablt_id,
                     'task_type':swim_type_list,
                     'swim_length':swim_len_list,
                     'swim_power':swim_power_list,
                     'swim_sig':swim_sig}
    fish_behavior = pd.DataFrame.from_dict(fish_behavior)


    mean_data = pd.DataFrame(fish_behavior.groupby(['fish_id', 'ablt_id'])['swim_sig', 'swim_power', 'swim_length'].mean()).reset_index()
    valid_list = np.ones(len(mean_data)).astype('bool')
    for _, n_fish in mean_data.iterrows():
        if sum(mean_data['fish_id']==n_fish['fish_id']) !=2:
            valid_list[_] = False
    mean_data = mean_data[valid_list]
    print(f'number of fish: {valid_list.sum()}')
    before_ = mean_data[mean_data['ablt_id']=='before']['swim_sig'].values*gain_stat_len/300
    after_ = mean_data[mean_data['ablt_id']=='after']['swim_sig'].values*gain_stat_len/300
    plt.plot(['before', 'after'], [before_, after_], '-ok', lw=2, ms=15)
    plt.xticks(rotation=-80)
    plt.ylabel('Period of signficance (sec)')
    sns.despine()
    plt.savefig(f"../Plots/gain_ablt/gain_sig_ablt{c_str}.pdf")
    plt.close('all')


    before_ = mean_data[mean_data['ablt_id']=='before']['swim_power'].values
    after_ = mean_data[mean_data['ablt_id']=='after']['swim_power'].values
    plt.plot(['before', 'after'], [before_/before_, after_/before_], '-ok', lw=2, ms=15)
    plt.xticks(rotation=-80)
    plt.ylabel('Normalized swim power')
    sns.despine()
    plt.savefig(f"../Plots/gain_ablt/swim_power_ablt{c_str}.pdf")
    plt.close('all')


    before_ = mean_data[mean_data['ablt_id']=='before']['swim_length'].values
    after_ = mean_data[mean_data['ablt_id']=='after']['swim_length'].values
    plt.plot(['before', 'after'], [before_/before_, after_/before_], '-ok', lw=2, ms=15)
    plt.xticks(rotation=-80)
    plt.ylabel('Normalized swim length')
    sns.despine()
    plt.savefig(f"../Plots/gain_ablt/swim_length_ablt{c_str}.pdf")
    plt.close('all')
    
    
behave_compare(control_=True, c_str='_control')
behave_compare(control_=False, c_str='')