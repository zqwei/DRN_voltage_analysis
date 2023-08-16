from utils import *
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.optimize import nnls
from scipy.optimize import minimize, LinearConstraint, Bounds


def sigmoid(x):
    return np.where(x>=0, 1/(1 + np.exp(-x)), np.exp(x)/(1 + np.exp(x)))


def log_sig(x):
    return np.where(x>= 0, -np.log(1+np.exp(-x)), x-np.log(1+np.exp(x)))


def nll_sig(w, X, y, w0, k):
    q = 2*y-1
    t = np.dot(X,w).squeeze()
    t = w0+k*t
    return -np.sum(log_sig(q*t))


def nllgrad_sig(w, X, y, w0, k):
    t = np.dot(X,w).squeeze()
    t = w0+k*t
    L = sigmoid(t)
    return -np.dot(y-L,X)*k


def nll_linear(w, X, y):
    return ((y-X.dot(w))**2).sum()


def nllgrad_linear(w, X, y):
    return -2*(y-X.dot(w)).dot(X)


## regularization ====
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


## total nll
def nll(w, X_dat, Y_dat, Y_dat_, ind_, wh=(0, 1, 0.3), reg=0.3):
    return nll_sig(w, X_dat, Y_dat, wh[0], wh[1])+wh[2]*nll_linear(w, X_dat, Y_dat_)+reg*regFunc(w, ind_)


## total nll grad
def nllgrad(w, X_dat, Y_dat, Y_dat_, ind_, wh=(0, 1, 0.3), reg=0.3):
    return nllgrad_sig(w, X_dat, Y_dat, wh[0], wh[1])+wh[2]*nllgrad_linear(w, X_dat, Y_dat_)+reg*regFuncgrad(w, ind_)


# # total nll
# def nll(w, X_dat, Y_dat, Y_dat_, ind_, wh=(0, 1, 0.3), reg=0.3):
#     return nll_linear(w, X_dat, Y_dat_)+reg*regFunc(w, ind_)


# ## total nll grad
# def nllgrad(w, X_dat, Y_dat, Y_dat_, ind_, wh=(0, 1, 0.3), reg=0.3):
#     return nllgrad_linear(w, X_dat, Y_dat_)+reg*regFuncgrad(w, ind_)


def NNLR_w(X_dat, Y_dat, Y_dat_, ind_, w0=None, wh=(0, 1, 0.3), reg=0.3):
    res = minimize(nll, w0, args=(X_dat, Y_dat, Y_dat_,ind_,wh,reg), method='L-BFGS-B', \
                   jac=nllgrad, options={'disp':False})
    return w0, res.x


def ll_func(w, X_dat, Y_dat, Y_dat_, wh=(0, 1, 0.3)):
    return -(nll_sig(w, X_dat, Y_dat, wh[0], wh[1])+wh[2]*nll_linear(w, X_dat, Y_dat_)+np.log(1/wh[2]*np.pi)/2*len(Y_dat_))


