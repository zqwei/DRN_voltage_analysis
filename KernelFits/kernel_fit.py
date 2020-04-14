from utils import *
from sklearn.linear_model import LogisticRegression
from scipy.optimize import nnls
from scipy.optimize import minimize, LinearConstraint, Bounds


def sigmoid(x):
    return np.where(x>=0, 1/(1 + np.exp(-x)), np.exp(x)/(1 + np.exp(x)))


def log_sig(x):
    return np.where(x>= 0, -np.log(1+np.exp(-x)), x-np.log(1+np.exp(x)))


def nll_sig(w, X, y):
    q = 2*y-1
    return -np.sum(log_sig(q*np.dot(X,w).squeeze()))


def nllgrad_sig(w, X, y):
    L = sigmoid(np.dot(X,w))
    return -np.dot(y-L,X)


# def nll_linear(w, s, X, y): #s = sigma^2
#     return ((y-X.dot(w))**2).sum()/(2*s)+np.log(2*np.pi*s)/2*len(y)


# def nllgrad_linear(w, s, X, y):
#     return -1/s*(y-X.dot(w)).dot(X)


# def nllgrad_linear_sigma(w, s, X, y):
#     return len(y)/2/s-1/2/s/s*((y-X.dot(w))**2).sum()


def nll_linear(w, X, y): #s = sigma^2
    return ((y-X.dot(w))**2).sum()


def nllgrad_linear(w, X, y):
    return -2*(y-X.dot(w)).dot(X)


def diffM(n):
    D=np.eye(n)+np.diag(-np.ones(n-1),k=1)
    D=D[:n-1]
    return D.T.dot(D)


def diff_l2(w):
    return w.T.dot(diffM(len(w))).dot(w)


def diff_l2_grad(w):
    return 2*(diffM(len(w))).dot(w)


def regFunc(w, ind_):
    num_ind_=len(ind_)
    return np.sum([diff_l2(w[ind_[_]:ind_[_+1]]) for _ in range(num_ind_-1)])


def regFuncgrad(w, ind_):
    num_ind_=len(ind_)
    grad_ = np.zeros(len(w))
    for _ in range(num_ind_-1):
        grad_[ind_[_]:ind_[_+1]] = diff_l2_grad(w[ind_[_]:ind_[_+1]])
    return grad_


# def nll(w, X_dat, Y_dat, Y_dat_, ind_, wl=0.3, reg=0.3):
#     return nll_sig(w[:-1], X_dat, Y_dat)+wl*nll_linear(w[:-1], w[-1], X_dat, Y_dat_)+reg*regFunc(w[:-1], ind_)


# def nllgrad(w, X_dat, Y_dat, Y_dat_, ind_, wl=0.3, reg=0.3):
#     grad_=w.copy()
#     grad_[:-1]=nllgrad_sig(w[:-1], X_dat, Y_dat)+wl*nllgrad_linear(w[:-1], w[-1], X_dat, Y_dat_)+reg*regFuncgrad(w[:-1], ind_)
#     grad_[-1]=nllgrad_linear_sigma(w[:-1], w[-1], X_dat, Y_dat_)
#     return grad_


def nll(w, X_dat, Y_dat, Y_dat_, ind_, wl=0.3, reg=0.3):
    return nll_sig(w, X_dat, Y_dat)+wl*nll_linear(w, X_dat, Y_dat_)+reg*regFunc(w, ind_)


def nllgrad(w, X_dat, Y_dat, Y_dat_, ind_, wl=0.3, reg=0.3):
    return nllgrad_sig(w, X_dat, Y_dat)+wl*nllgrad_linear(w, X_dat, Y_dat_)+reg*regFuncgrad(w, ind_)


def NNLR_w(X_dat, Y_dat, Y_dat_, ind_, w0=None, wl=0.3, reg=0.3):
    lb = np.zeros(len(w0))
    ub = np.zeros(len(w0))
    ub[:] = np.inf
    lb[:ind_[0]] = -np.inf
#     lb[ind_[0]:ind_[0]+visu_pad//2]=0
#     ub[ind_[0]:ind_[0]+visu_pad//2]=0
    res = minimize(nll, w0, args=(X_dat, Y_dat, Y_dat_,ind_,wl,reg), method='L-BFGS-B', \
                   jac=nllgrad, bounds=Bounds(lb, ub), options={'disp':False}) # 
    return w0, res.x


def ll_func(w, X_dat, Y_dat, Y_dat_, wl=1):
    return -(nll_sig(w, X_dat, Y_dat)+wl*nll_linear(w, X_dat, Y_dat_)+np.log(1/wl*np.pi)/2*len(Y_dat_))


def nll_null(w, X_dat, Y_dat, Y_dat_, wl=0.3):
    return nll_sig(w, X_dat, Y_dat)+wl*nll_linear(w, X_dat, Y_dat_)


def nllgrad_null(w, X_dat, Y_dat, Y_dat_, wl=0.3):
    return nllgrad_sig(w, X_dat, Y_dat)+wl*nllgrad_linear(w, X_dat, Y_dat_)


def NNLR_full_w(X_dat, Y_dat, Y_dat_, w0=None, wl=0.3):
    res = minimize(nll_null, w0, args=(X_dat, Y_dat, Y_dat_, wl), method='L-BFGS-B', \
                   jac=nllgrad_null, options={'disp':True}) # 
    return w0, res.x