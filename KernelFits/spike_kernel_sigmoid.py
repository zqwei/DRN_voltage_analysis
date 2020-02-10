import numpy as np
from sklearn.linear_model import LogisticRegression


def model_fit(X_dat, Y_dat):    
    lr = LogisticRegression(fit_intercept=False).fit(X_dat,Y_dat)
    return lr.coef_


def sigmoid(x):
    return np.where(x>=0, 1/(1 + np.exp(-x)), np.exp(x)/(1 + np.exp(x)))


def log_sig(x):
    return np.where(x>= 0, -np.log(1+np.exp(-x)), x-np.log(1+np.exp(x)))


def loglike(X, y, w):
    q = 2*y-1
    return np.sum(log_sig(q*np.dot(X,w).squeeze()))


def loglike_grad(X, y, w):
    L = sigmoid(np.dot(X,w))
    return np.dot(y - L,X)


def loglike_null(X, y, w):
    q = 2*y-1
    return np.sum(log_sig(q*X*w))


def loglike_hessian(X, y, w):
    L = sigmoid(np.dot(X,w))
    return -np.dot(L*(1-L)*X.T,X)