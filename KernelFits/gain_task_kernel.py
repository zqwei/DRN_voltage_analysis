from utils import *
from kernel_fit import *


def kernel_fit(sub_list, behavior_dat, pad_list, trial_valid_fit, reg=100000, t_pre=100):
    Y_dat = []
    Y_dat_ = []
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
            if n_time>visu_pad:
                spk_history=spk_[n_time-visu_pad:n_time]
                visu_history=visu_[n_time-visu_pad:n_time]
                swim_history=swim_[n_time-visu_pad:n_time+swim_pad]
            else:
                visu_history=np.zeros(visu_pad)
                spk_history=np.zeros(visu_pad)
                swim_history=np.zeros(visu_pad+swim_pad)
                if n_time>t_pre:
                    visu_history[-n_time:]=visu_[:n_time]
                    spk_history[-n_time:]=spk_[:n_time]
                    swim_history[-(n_time+swim_pad):]=swim_[:n_time+swim_pad]
            Y_dat.append(spk_[n_time])
            Y_dat_.append(dff_[n_time])
            X_dat.append(np.r_[1, dff_[n_time-visu_pad], -spk_history, np.sqrt(swim_history)/100, np.sqrt(visu_history)/100])

    Y_dat = np.array(Y_dat)
    X_dat = np.array(X_dat)
    Y_dat_ = np.array(Y_dat_)
    ind_=[2, visu_pad, visu_pad+swim_pad, visu_pad]
    ind_=np.cumsum(ind_)
    w=[]
    ll_model=[]
    comp = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for sig_i, sig_j in comp:
        X_dat_ = X_dat.copy()
        X_dat_[:, ind_[1]:ind_[2]]=sig_i*X_dat_[:, ind_[1]:ind_[2]]
        X_dat_[:, ind_[2]:ind_[3]]=sig_j*X_dat_[:, ind_[2]:ind_[3]]
        lr = LogisticRegression(penalty='l2',fit_intercept=False, C=1, solver='lbfgs', max_iter=5000).fit(X_dat_, Y_dat)
        w0 = lr.coef_.copy().T
        nn_indx = w0<0
        nn_indx[:ind_[0], 0]=False
        w0[nn_indx]=0
        s = Y_dat_.var()
        # alternating optimization
        for n in range(10):
            w0, w_ = NNLR_w(X_dat_, Y_dat, Y_dat_, ind_, w0=w0, wl=1/s, reg=reg)
            s=((Y_dat_-X_dat_.dot(w_))**2).mean()
            w0 = w_.copy()
        w.append(w_)
        ll_model.append(ll_func(w_, X_dat_, Y_dat, Y_dat_, wl=1/s))
    n_model=np.argmax(ll_model)
    sig_i, sig_j = comp[n_model]
    X_dat_ = X_dat.copy()
    X_dat_[:, ind_[1]:ind_[2]]=sig_i*X_dat_[:, ind_[1]:ind_[2]]
    X_dat_[:, ind_[2]:ind_[3]]=sig_j*X_dat_[:, ind_[2]:ind_[3]]
    w_ = w[n_model]
    s=((Y_dat_-X_dat_.dot(w_))**2).mean()
    ll_model_ = ll_func(w_, X_dat_, Y_dat, Y_dat_, wl=1/s)
    
#     w_new=np.zeros(w_.shape)
#     w_new[ind_[0]:ind_[1]]=w_[ind_[0]:ind_[1]]
#     w_new[:ind_[0]]=w_[:ind_[0]]
    w_new=w_.copy()
    w_new[ind_[0]:ind_[1]]=0
    s=((Y_dat_-X_dat_.dot(w_new))**2).mean()
    ll_spike = ll_func(w_new, X_dat_, Y_dat, Y_dat_, wl=1/s)
    
#     w_new=np.zeros(w_.shape)
#     w_new[ind_[1]:ind_[2]]=w_[ind_[1]:ind_[2]]
#     w_new[:ind_[0]]=w_[:ind_[0]]
    w_new=w_.copy()
    w_new[ind_[1]:ind_[2]]=0
    s=((Y_dat_-X_dat_.dot(w_new))**2).mean()
    ll_swim = ll_func(w_new, X_dat_, Y_dat, Y_dat_, wl=1/s)
    
#     w_new=np.zeros(w_.shape)
#     w_new[ind_[2]:ind_[3]]=w_[ind_[2]:ind_[3]]
#     w_new[:ind_[0]]=w_[:ind_[0]]
    w_new=w_.copy()
    w_new[ind_[2]:ind_[3]]=0
    s=((Y_dat_-X_dat_.dot(w_new))**2).mean()
    ll_visual = ll_func(w_new, X_dat_, Y_dat, Y_dat_, wl=1/s)
    return (w_, w), (ll_model, ll_model_, ll_spike, ll_swim, ll_visual), comp[n_model]

