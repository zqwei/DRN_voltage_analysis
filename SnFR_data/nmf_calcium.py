import numpy as np


def denoise_sig(M):
    import mkl
    mkl.set_num_threads(4)
    import numpy as np
    from sklearn.utils.extmath import randomized_svd
    dimsM = M.shape
    M = M.reshape((np.prod(dimsM[:-1]),dimsM[-1]),order='F')
    k = min(min(M.shape)//4, 600)
    # [U, S, Va] = pca(M.T, k=k, n_iter=20, raw=True)
    [U, S, Va] = randomized_svd(M.T, k, n_iter=10, power_iteration_normalizer='QR')
    M_pca = U.dot(np.diag(S).dot(Va))
    return M_pca.T.reshape(dimsM, order='F'), U, S, Va, dimsM


def demix_components(M, save_root=''):
    import mkl
    mkl.set_num_threads(4)
    import numpy as np
    from fish_proc.demix import superpixel_analysis as sup
    from fish_proc.utils.snr import local_correlations_fft
    from fish_proc.utils.demix import recompute_C_matrix
    import pickle
    
    Cn_ = local_correlations_fft(M)
    cut_off_point = np.percentile(Cn_[:], [95, 80, 70, 50])
    pass_num = 4
    pass_num_max = 4
    is_demix = False
    while not is_demix and pass_num>=0:
        try:
            rlt_= sup.demix_whole_data(M, cut_off_point[pass_num_max-pass_num:], length_cut=[20, 20, 40, 40],
                                       th=[1,1,1,1], pass_num=pass_num, residual_cut = [0.6,0.6,0.6,0.6],
                                       corr_th_fix=0.3, max_allow_neuron_size=0.3, merge_corr_thr=0.90,
                                       merge_overlap_thr=0.6, num_plane=1, patch_size=[10, 10], plot_en=False,
                                       TF=False, fudge_factor=1, text=False, bg=False, max_iter=50,
                                       max_iter_fin=90, update_after=40)
            is_demix = True
        except:
            print(f'fail at pass_num {pass_num}', flush=True)
            is_demix = False
            pass_num -= 1
            
    A_ = rlt_['fin_rlt']['a']
    C_ = recompute_C_matrix(M, A_)
    
    with open(save_root+'/components_rlt.pkl', 'wb') as f:
            pickle.dump(rlt_, f)
    np.savez(save_root+'/components.npz', A_=A_, C_=C_)
    
    return None


def baseline(data, window=100, percentile=15, downsample=1, axis=-1):
    """
    Get the baseline of a numpy array using a windowed percentile filter with optional downsampling
    data : Numpy array
        Data from which baseline is calculated
    window : int
        Window size for baseline estimation. If downsampling is used, window shrinks proportionally
    percentile : int
        Percentile of data used as baseline
    downsample : int
        Rate of downsampling used before estimating baseline. Defaults to 1 (no downsampling).
    axis : int
        For ndarrays, this specifies the axis to estimate baseline along. Default is -1.
    """
    from scipy.ndimage.filters import percentile_filter
    from scipy.interpolate import interp1d
    from numpy import ones

    size = ones(data.ndim, dtype='int')
    size[axis] *= window//downsample

    if downsample == 1:
        bl = percentile_filter(data, percentile=percentile, size=size)
    else:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, None, downsample)
        data_ds = data[slices]
        baseline_ds = percentile_filter(data_ds, percentile=percentile, size=size)
        interper = interp1d(range(0, data.shape[axis], downsample), baseline_ds, axis=axis, fill_value='extrapolate')
        bl = interper(range(data.shape[axis]))
    return bl


def baseline_correct(block_b, block_t):
    min_t = np.percentile(block_t, 0.3, axis=-1, keepdims=True)
    min_t[min_t>0] = 0
    min_b = np.min(block_b-min_t, axis=-1, keepdims=True)
    min_b[min_b<=0] = min_b[min_b<=0] - 0.01
    min_b[min_b>0] = 0
    return block_b - min_t - min_b