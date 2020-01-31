from utils import *

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

# for ind, row in dat_xls_file.iterrows():
# folder = row['folder']
# fish = row['fish']
# _ = np.load(f'../Analysis/swim_power/{folder}_{fish}_swim_dat.npz')
# swim_starts = _['swim_starts']
# swim_ends = _['swim_ends']
# r_swim = _['r_swim']
# l_swim = _['l_swim']
# visu = _['visu']
# task_period = _['task_period']
# swim_task_index = _['swim_task_index']

# trial_valid = np.ones(len(swim_starts)).astype('bool')
# for n, n_swim in enumerate(swim_starts[:-1]):        
#     # examine the swim with short inter-swim-interval
#     if swim_starts[n+1] - n_swim < t_sig:    
#         trial_valid[n] = False
# p_swim = l_swim+r_swim
# trial_valid_fit = trial_valid & ((visu[:,:t_pre-10]<=0).sum(axis=-1)==0)

# X_dat = []

# for n_, dff_ in enumerate(sub_list):
#     if not trial_valid_fit[n_]:
#         continue
#     swim_ = p_swim[n_]
#     visu_ = -visu[n_]
#     visu_[visu_<0]=0
#     for n_time in range(t_pre, t_pre+240):
#         swim_history=swim_[n_time-swim_pad:n_time+swim_pad]
#         if n_time>120:
#             visu_history=visu_[n_time-visu_pad:n_time]
#         else:
#             visu_history=np.zeros(visu_pad)
#         Y_dat.append(dff_[n_time]-dff_[n_time-30])
#         X_dat.append(np.r_[-spk_history, -np.sqrt(swim_history)/100, np.sqrt(visu_history)/100])




# ncell_ = 6
# _ = np.load(f'../Analysis/swim_voltr/{folder}_{fish}_swim_voltr_dat.npz')
# # trial_valid = _['trial_valid']
# sub_swim = _['sub_swim'][ncell_]
# spk_list = _['raw_spk_swim'][ncell_]


def subvolt_fit(sub_list, behavior_dat, spike_pad=spike_pad, swim_pad=swim_pad, visu_pad=visu_pad):
    Y_dat = []
    X_dat = []
    for n_, dff_ in enumerate(sub_list):
        if not trial_valid_fit[n_]:
            continue
        spk_ = spk_list[n_]
        swim_ = p_swim[n_]
        visu_ = -visu[n_]
        visu_[visu_<0]=0
        for n_time in range(t_pre, t_pre+240):
            spk_history=spk_[n_time-spike_pad:n_time]
            swim_history=swim_[n_time-swim_pad:n_time+swim_pad]
            if n_time>120:
                visu_history=visu_[n_time-visu_pad:n_time]
            else:
                visu_history=np.zeros(visu_pad)
            Y_dat.append(dff_[n_time]-dff_[n_time-30])
            X_dat.append(np.r_[-spk_history, -np.sqrt(swim_history)/100, np.sqrt(visu_history)/100])

    Y_dat = np.array(Y_dat)
    X_dat = np.array(X_dat)

    w, rnorm = nnls(X_dat, Y_dat)

    ev_model = explained_variance(Y_dat,X_dat.dot(w))
    ev_spike = explained_variance(Y_dat,[:,:spike_pad].dot(w[:spike_pad]))
    ev_swim = explained_variance(Y_dat,X_dat[:,spike_pad:(spike_pad+swim_pad*2)].dot(w[spike_pad:(spike_pad+swim_pad*2)]))
    ev_visual = explained_variance(Y_dat,X_dat[:,(spike_pad+swim_pad*2):].dot(w[(spike_pad+swim_pad*2):]))


if __name__ == "__main__":
    subvolt_fit(None, None)