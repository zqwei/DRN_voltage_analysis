from utils import *
from kernel_fit import *
from utils import explained_variance
from spike_kernel_sigmoid import model_fit as spk_fit
from spike_kernel_sigmoid import loglike, loglike_null


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
        for n_time in range(t_pre, t_pre+350):
            spk_history=spk_[n_time-spike_pad:n_time]
            swim_history=swim_[n_time-swim_pad:n_time]
            if n_time>visu_pad:
                spk_history=spk_[n_time-visu_pad:n_time]
                visu_history=visu_[n_time-visu_pad:n_time]
                swim_history=swim_[n_time-visu_pad:n_time]
            else:
                visu_history=np.zeros(visu_pad)
                spk_history=np.zeros(visu_pad)
                swim_history=np.zeros(visu_pad)
                if n_time>t_pre:
                    visu_history[-n_time:]=visu_[:n_time]
                    spk_history[-n_time:]=spk_[:n_time]
                    swim_history[-(n_time):]=swim_[:n_time]
            Y_dat.append(spk_[n_time])
            Y_dat_.append(dff_[n_time])
            X_dat.append(np.r_[1, dff_[n_time-visu_pad], spk_history, np.sqrt(swim_history)/1000, np.sqrt(visu_history)/1000])

    Y_dat = np.array(Y_dat)
    X_dat = np.array(X_dat)
    Y_dat_ = np.array(Y_dat_)
    ind_=[2, visu_pad, visu_pad, visu_pad]
    ind_=np.cumsum(ind_)
    lr = LogisticRegression(penalty='l2',fit_intercept=False, C=1, solver='lbfgs', max_iter=5000).fit(X_dat, Y_dat)
    w0 = lr.coef_.copy().T
    s = Y_dat_.var()
    c = 0
    k = 1
    # alternating optimization
    for n in range(5):
        w0, w_ = NNLR_w(X_dat, Y_dat, Y_dat_, ind_, w0=w0, wh=[c, k, 1/s], reg=100000)
        s=((Y_dat_-X_dat.dot(w_))**2).mean()
        lr = LogisticRegression(penalty='l2',fit_intercept=True, C=1000, solver='lbfgs', max_iter=5000).fit(X_dat.dot(w_)[:, None], Y_dat)
        k = lr.coef_[0]
        c = lr.intercept_[0]
        w0 = w_.copy()
    ev_model = explained_variance(Y_dat_,X_dat.dot(w_))
    w_null = spk_fit(X_dat[:,0][:, np.newaxis], Y_dat)
    ll_full = loglike(X_dat.dot(w_)*k+c, Y_dat, 1)
    ll_null = loglike_null(X_dat[:,0], Y_dat, w_null)
    return w_, ev_model, 1-ll_full/ll_null,

