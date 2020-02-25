from utils import *


vol_file = '../Analysis/depreciated/analysis_sections_ablation_gain.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
t_len = t_pre+t_post
t_sig = 300 # time used for significance test after swim
non_spike_thres = 100
k_spk = boxcarKernel(sigma=61)
k_sub = 10


def cell_shape_check(control_=False, c_str=''):
    cell_shape = []
    plot_ = True
    spk_thres = 60

    for ind, row in dat_xls_file.iterrows():
        ablation_pair = search_paired_data(row, dat_xls_file)

        if not ablation_pair:
            continue

        folder = row['folder']
        fish = row['fish'][:-6]
        task_type = row['task']

        if ('control' in task_type)==control_:
            continue    

        dir_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'
        dat_dir = dir_folder+f'{folder}/{fish}before/Data/'
        if not os.path.exists(dir_folder+f'{folder}/{fish}before/Data/'+'Voltr_spikes.npz'):
            continue
        if not os.path.exists(dir_folder+f'{folder}/{fish}after/Data/'+'Voltr_spikes.npz'):
            continue

        dff = np.load(dat_dir+'Voltr_spikes.npz')['voltrs']
        spk = np.load(dat_dir+'Voltr_spikes.npz')['spk']
        dff = dff - np.nanmedian(dff, axis=1, keepdims=True)
        num_cell = spk.shape[0]
        spk = np.r_['-1', np.zeros((num_cell, 600)), spk]

        dat_dir = dir_folder+f'{folder}/{fish}after/Data/'
        dff_ = np.load(dat_dir+'Voltr_spikes.npz')['voltrs']
        spk_ = np.load(dat_dir+'Voltr_spikes.npz')['spk']
        dff_ = dff_ - np.nanmedian(dff_, axis=1, keepdims=True)
        num_cell = spk_.shape[0]
        spk_ = np.r_['-1', np.zeros((num_cell, 600)), spk_]

        for n_cell in range(num_cell):
            if (spk[n_cell].sum()<spk_thres): # or (spk_[n_cell].sum()<spk_thres)
                continue
            spk_shape_b = spk_shape(np.where(spk[n_cell])[0], dff[n_cell])
            spk_shape_a = spk_shape(np.where(spk_[n_cell])[0], dff_[n_cell])        
            var_ = min(spk_shape_b.std(), spk_shape_a.std())
            err_ = np.sqrt(((spk_shape_a-spk_shape_b)**2).mean())
            if not np.isnan(err_/var_):
                cell_shape.append(err_/var_)
            if plot_ and (err_/var_<0.2):
                plt.figure(figsize=(8, 6))
                plt.plot(spk_shape_b)
                plt.plot(spk_shape_a)
                var_ = min(spk_shape_b.std(), spk_shape_a.std())
                err_ = np.sqrt(((spk_shape_a-spk_shape_b)**2).mean())
                plt.title(err_/var_)
                plt.savefig(f'../Plots/gain_ablt/cell_shape_ablt_exp.pdf')
                plt.close('all')
                plot_=False
    
    plt.figure(figsize=(8, 6))
    sns.distplot(np.array(cell_shape)**2, hist=True, rug=True, bins=np.arange(0, 10, 0.1))
    plt.ylabel('Prob density')
    plt.xlabel('Relative difference')
    plt.xlim([0, 2])
    sns.despine()
    plt.savefig(f'../Plots/gain_ablt/cell_shape_ablt{c_str}.pdf')
    plt.close('all')
    
cell_shape_check(control_=True, c_str='_control')
cell_shape_check(control_=False, c_str='')