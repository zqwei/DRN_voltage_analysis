from utils import *
from gain_task_spike_kernel_sigmoid import spike_fit

vol_file = '../Analysis/depreciated/analysis_sections_gain.csv'
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_xls_file = dat_xls_file.reset_index()
t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
t_label = np.arange(-t_pre, t_post)/300
t_sig = 240
sub_pad = 9
visu_pad = 200

wfull_list=[]
wsub_list=[]
w0_list=[]
ev_full_list=[]
ev_sub_list=[]

for ind, row in dat_xls_file.iterrows():
    folder = row['folder']
    fish = row['fish']
    print(f'{folder}_{fish}')
    _ = np.load(f'../Analysis/swim_power/{folder}_{fish}_swim_dat.npz')
    swim_starts = _['swim_starts']
    swim_ends = _['swim_ends']
    r_swim = _['r_swim']
    l_swim = _['l_swim']
    visu = _['visu']
    task_period = _['task_period']
    swim_task_index = _['swim_task_index']

    trial_valid = np.ones(len(swim_starts)).astype('bool')
    for n, n_swim in enumerate(swim_starts[:-1]):        
        # examine the swim with short inter-swim-interval
        if swim_starts[n+1] - n_swim < t_sig:    
            trial_valid[n] = False
    p_swim = l_swim+r_swim
    trial_valid_fit = trial_valid & ((visu[:,:t_pre-10]<=0).sum(axis=-1)==0)
    
    _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}_swim_voltr_dat.npz')
    
    sub_swim_list = _['sub_swim']
    spk_list_list = _['raw_spk_swim']
    
    num_cell = len(sub_swim_list)
    if num_cell==0:
        continue
    
    for ncell_ in range(num_cell):
        print(ncell_)
        sub_swim = sub_swim_list[ncell_]
        spk_list = spk_list_list[ncell_]
        sub_list = sub_swim-sub_swim[:, 70:75].mean(axis=-1, keepdims=True)
        pad_list = [sub_pad, visu_pad]
        w_all, ev_all = spike_fit(spk_list, sub_list, visu, pad_list, trial_valid_fit, reg=3, t_pre=100)
        w_full, w_null, w_sub = w_all
        ev_full, ev_sub = ev_all
        wfull_list.append(w_full)
        wsub_list.append(w_sub)
        w0_list.append(w_null)
        ev_full_list.append(ev_full)
        ev_sub_list.append(ev_sub)

np.savez('depreciated/spike_fit.npz', wfull_list=np.array(wfull_list), \
         wsub_list=np.array(wsub_list), \
         w0_list=np.array(w0_list), \
         ev_full_list=np.array(ev_full_list), \
         ev_sub_list=np.array(ev_sub_list))

