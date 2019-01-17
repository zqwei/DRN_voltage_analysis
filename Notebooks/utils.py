import numpy as np

def smooth(a, kernel):
    b=np.convolve(a,kernel,'same')/np.convolve(np.ones(a.shape),kernel,'same')
    return b

def gaussKernel(sigma=20):
    return (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(np.arange(-sigma*3,sigma*3+1)**2)/(2*sigma**2))

def bootstrap_p_ABtest(test, ctrl, is_left=True):
    import bootstrapped.bootstrap as bs
    import bootstrapped.stats_functions as bs_stats
    import bootstrapped.compare_functions as bs_comp
    test_ctrl_dist = bs.bootstrap_ab(test, ctrl, bs_stats.mean, bs_comp.difference, return_distribution=True)
    if is_left:
        return (test_ctrl_dist<0).mean()
    else:
        return (test_ctrl_dist>0).mean()