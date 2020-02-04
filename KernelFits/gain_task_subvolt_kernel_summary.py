from utils import *
from gain_task_subvolt_kernel import subvolt_fit

vol_file = '../Analysis/depreciated/analysis_sections_gain.csv'
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_xls_file = dat_xls_file.reset_index()
t_pre = 100 # time window pre-swim
t_post = 350 # time window post-swim
t_label = np.arange(-t_pre, t_post)/300
t_sig = 240
spike_pad = 2
swim_pad = 30
visu_pad = 60

w0_list=[]
w_list=[]
w_list_=[]
ev_model_list_=[]
ev_model_list=[]
comp_list=[]
ev_spike_list=[]
ev_swim_list=[]
ev_visual_list=[]


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
        behavior_dat = [spk_list, p_swim, visu]
        pad_list = [spike_pad, swim_pad, visu_pad]
        w_all, ev_all, comp = subvolt_fit(sub_list, behavior_dat, pad_list, trial_valid_fit, t_pre=t_pre, reg=3)
        w0, w, w_ = w_all
        ev_model_, ev_model, ev_spike, ev_swim, ev_visual = ev_all
        w0_list.append(w0)
        w_list.append(w)
        w_list_.append(w_)
        ev_model_list_.append(ev_model_)
        ev_model_list.append(ev_model)
        comp_list.append(comp)
        ev_spike_list.append(ev_spike)
        ev_swim_list.append(ev_swim)
        ev_visual_list.append(ev_visual)

np.savez('depreciated/subvolt_fit.npz', w0_list=np.array(w0_list), \
         w_list=np.array(w_list), \
         w_list_=np.array(w_list_), \
         ev_model_list_=np.array(ev_model_list_), \
         ev_model_list=np.array(ev_model_list), \
         comp_list=np.array(comp_list), \
         ev_spike_list=np.array(ev_spike_list), \
         ev_swim_list=np.array(ev_swim_list), \
         ev_visual_list=np.array(ev_visual_list))