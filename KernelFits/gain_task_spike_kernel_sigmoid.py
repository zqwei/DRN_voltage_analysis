from utils import *
from spike_kernel_sigmoid import model_fit, loglike, loglike_null


def spike_fit(spk_list, sub_list, visu, pad_list, trial_valid_fit, reg=3, t_pre=100):
    Y_dat = []
    X_dat = []
    sub_pad, visu_pad = pad_list
    for n_, dff_ in enumerate(sub_list):
        if not trial_valid_fit[n_]:
            continue
        spk_ = spk_list[n_]
        visu_ = -visu[n_]
        visu_[visu_<0]=0
        for n_time in range(t_pre, t_pre+240):
            if n_time>visu_pad:
                visu_history=visu_[n_time-visu_pad:n_time]
            else:
                visu_history=np.zeros(visu_pad)
                if n_time>t_pre:
                    visu_history[-n_time:]=visu_[:n_time]
            Y_dat.append(spk_[n_time])
            X_dat.append(np.r_[1, dff_[n_time-sub_pad:n_time],np.sqrt(visu_history)/100])

    Y_dat = np.array(Y_dat)
    X_dat = np.array(X_dat)
    
    w_full = model_fit(X_dat, Y_dat)
    w_null = model_fit(X_dat[:,0][:, np.newaxis], Y_dat)
    w_sub = model_fit(X_dat[:,:(sub_pad+1)], Y_dat)
    
    ll_full = loglike(X_dat, Y_dat, w_full.T)
    ll_null = loglike_null(X_dat[:,0], Y_dat, w_null)
    ll_sub = loglike(X_dat[:, :(sub_pad+1)], Y_dat, w_sub.T)
    
    print((1-ll_full/ll_null, 1-ll_sub/ll_null))
    
    return (w_full, w_null, w_sub), (1-ll_full/ll_null, 1-ll_sub/ll_null)

