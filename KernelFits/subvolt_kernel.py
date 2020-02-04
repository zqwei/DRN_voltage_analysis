import numpy as np
from scipy.optimize import nnls
from scipy.optimize import minimize, LinearConstraint, Bounds


def nonsmooth_w(X_dat, Y_dat):
    w0, _ = nnls(X_dat, Y_dat)
    return w0


def diff_l2(w):
    return ((w[:-1]-w[1:])**2).sum()


def regFunc(w, ind_):
    num_ind_=len(ind_)
    return np.sum([diff_l2(w[ind_[_]:ind_[_+1]]) for _ in range(num_ind_-1)])


def costFunc(w, r, ind_, X_dat, Y_dat):
    Y_hat = X_dat.dot(w)
    return ((Y_dat-Y_hat)**2).sum()+r*regFunc(w, ind_)


def diff_l2_w(X_dat, Y_dat, ind_, reg=3, w0=None):
    if w0 is None:
        w0 = nonsmooth_w(X_dat, Y_dat)
    lb = np.zeros(len(w0))
    ub = np.zeros(len(w0))
    ub[:] = np.inf
    res = minimize(costFunc, w0, args=(reg, ind_, X_dat, Y_dat), method='L-BFGS-B', bounds=Bounds(lb, ub), options={'disp':True})
    return w0, res.x