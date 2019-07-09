import numpy as np


def denoise_sig(M):
    from fbpca import pca
    dimsM = M.shape
    M = M.reshape((np.prod(dimsM[:-1]),dimsM[-1]),order='F')
    k = min(min(M.shape)//4, 600)
    [U, S, Va] = pca(M.T, k=k, n_iter=20, raw=True)
    M_pca = U.dot(np.diag(S).dot(Va))
    return M_pca.T.reshape(dimsM, order='F')


def demix_components(M, save_root=''):
    from fish_proc.demix import superpixel_analysis as sup
    from fish_proc.utils.snr import local_correlations_fft
    from fish_proc.utils.demix import recompute_C_matrix
    import pickle
    
    Cn_ = local_correlations_fft(M)
    cut_off_point = np.percentile(Cn_[:], [99, 95, 80, 60])
    pass_num = 4
    pass_num_max = 4
    is_demix = False
    while not is_demix and pass_num>=0:
        try:
            rlt_= sup.demix_whole_data(M, cut_off_point[pass_num_max-pass_num:], length_cut=[10, 10, 20, 20],
                                       th=[1,1,1,1], pass_num=pass_num, residual_cut = [0.6,0.6,0.6,0.6],
                                       corr_th_fix=0.3, max_allow_neuron_size=0.05, merge_corr_thr=0.90,
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
    
    with open(save_root+'components_rlt.pkl', 'wb') as f:
            pickle.dump(rlt_, f)
    np.savez(save_root+'components.npz', A_=A_, C_=C_)
    
    return None