import numpy as np
import matplotlib.pyplot as plt

def smooth(a, kernel):
    b=np.convolve(a,kernel,'same')/np.convolve(np.ones(a.shape),kernel,'same')
    return b

def gaussKernel(sigma=20):
    return (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(np.arange(-sigma*3,sigma*3+1)**2)/(2*sigma**2))

def bootstrap_p_ABtest(test, ctrl, is_left=True):
    import bootstrapped.bootstrap as bs
    import bootstrapped.stats_functions as bs_stats
    import bootstrapped.compare_functions as bs_comp
    num_ = 1000 # use for Multi-test correction
    p_val = 0.05
    boot_strp_iter = max(int(num_/p_val)*10+10, 10000)
    num_thread = 30
    test_ctrl_dist = bs.bootstrap_ab(test, ctrl, bs_stats.mean, bs_comp.difference, return_distribution=True, num_threads=num_thread, num_iterations=boot_strp_iter)
    if is_left:
        return (test_ctrl_dist<0).mean()
    else:
        return (test_ctrl_dist>0).mean()
    
def cont_mode(data, isplot=False):
    from scipy import stats
    kde = stats.gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 100)
    p = kde(x)
    if isplot:
        plt.plot(x, p)
    return x[np.argmax(p)]