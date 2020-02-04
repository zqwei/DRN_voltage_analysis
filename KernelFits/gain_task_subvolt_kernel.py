from utils import *
from subvolt_kernel import diff_l2_w, nonsmooth_w


def subvolt_fit(sub_list, behavior_dat, pad_list, trial_valid_fit, reg=3, t_pre=100):
    Y_dat = []
    X_dat = []
    spk_list, p_swim, visu = behavior_dat
    spike_pad, swim_pad, visu_pad = pad_list
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
            X_dat.append(np.r_[-spk_history, np.sqrt(swim_history)/100, np.sqrt(visu_history)/100])

    Y_dat = np.array(Y_dat)
    X_dat = np.array(X_dat)
    ind_=np.array([0, spike_pad, swim_pad*2, visu_pad])
    ind_=np.cumsum(ind_)
    w=[]
    ev_model=[]
    comp = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for sig_i, sig_j in comp:
        X_dat_ = X_dat.copy()
        X_dat_[:, ind_[1]:ind_[2]]=sig_i*X_dat_[:, ind_[1]:ind_[2]]
        X_dat_[:, ind_[2]:ind_[3]]=sig_j*X_dat_[:, ind_[2]:ind_[3]]
        w_ = nonsmooth_w(X_dat_, Y_dat)
        w.append(w_)
        ev_model.append(explained_variance(Y_dat,X_dat_.dot(w_)))
    n_model=np.argmax(ev_model)
    sig_i, sig_j = comp[n_model]
    X_dat_ = X_dat.copy()
    X_dat_[:, ind_[1]:ind_[2]]=sig_i*X_dat_[:, ind_[1]:ind_[2]]
    X_dat_[:, ind_[2]:ind_[3]]=sig_j*X_dat_[:, ind_[2]:ind_[3]]
    w0, w_ = diff_l2_w(X_dat_, Y_dat, ind_, reg=reg, w0=w[n_model])
    ev_model_ = explained_variance(Y_dat,X_dat_.dot(w_))
    ev_spike = explained_variance(Y_dat,X_dat_[:,:spike_pad].dot(w_[:spike_pad]))
    ev_swim = explained_variance(Y_dat,X_dat_[:,spike_pad:(spike_pad+swim_pad*2)].dot(w_[spike_pad:(spike_pad+swim_pad*2)]))
    ev_visual = explained_variance(Y_dat,X_dat_[:,(spike_pad+swim_pad*2):].dot(w_[(spike_pad+swim_pad*2):]))
    return (w0, w_, w), (ev_model, ev_model_, ev_spike, ev_swim, ev_visual), comp[n_model]

