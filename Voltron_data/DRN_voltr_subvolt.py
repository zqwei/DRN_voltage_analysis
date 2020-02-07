#!/groups/ahrens/home/weiz/miniconda3/envs/trefide/bin/python

import numpy as np
import pandas as pd
import os, sys
from trefide.temporal import TrendFilter
# this requires conda env trefide
dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/'


def tf_filter(_):
    from trefide.temporal import TrendFilter
    spk__, voltr_, voltr= _
    filters = TrendFilter(len(voltr_))
    tspk = np.where(spk__>0)[0]
    tspk_win = tspk[:, None] + np.arange(-3, 3)[None, :]
    tspk_win = tspk_win.reshape(-1)
    nospike = np.zeros(spk__.shape)
    nospike[tspk_win] = 1
    tspk_ = np.where(nospike==0)[0]
    int_voltr_ = voltr_.copy()
    int_voltr_[tspk_win] = np.interp(tspk_win, tspk_, voltr_[tspk_])
    denoised_voltr_ = filters.denoise(int_voltr_)
    int_voltr_ = voltr[600:].copy()
    int_voltr_[tspk_win] = np.interp(tspk_win, tspk_, voltr[600:][tspk_])
    denoised_voltr = filters.denoise(int_voltr_)
    out = (denoised_voltr_, denoised_voltr)
    return np.asarray(out[0])[np.newaxis,:], np.asarray(out[1])[np.newaxis,:]


def voltr2subvolt(row, fext=''):
    '''
    This one can be benefited from multiple cores.
    '''
    from pathlib import Path
    import multiprocessing as mp

    folder = row['folder']
    fish = row['fish']
    save_folder = dat_folder + f'{folder}/{fish}/Data'

    if os.path.isfile(save_folder+f'/finished_subvolt{fext}.tmp'):
        return None

    if not os.path.isfile(save_folder+f'/finished_spikes{fext}.tmp'):
        print('Spike file does not exist.')
        return None

    # if os.path.isfile(save_folder+f'/proc_subvolt{fext}.tmp'):
    #     print('SubVolt file is already in processing.')
    #     return None

    Path(save_folder+f'/proc_subvolt{fext}.tmp').touch()
    print(f'Processing {save_folder}')
    _ = np.load(f'{save_folder}/Voltr_spikes.npz')
    voltrs = _['voltrs']
    spk = _['spk']
    spkprob = _['spkprob']
    spk1 = _['spk1']
    spk2 = _['spk2']
    voltr_ = _['voltr_']
    n_, len_ = voltrs.shape
    spk__list = spk1+spk2
    dat_ = [(spk__list[_, :], voltr_[_, :], voltrs[_, :]) for _ in range(n_)]
    mp_count = min(mp.cpu_count(), n_)
    pool = mp.Pool(processes=mp_count)
    individual_results = pool.map(tf_filter, dat_)
    pool.close()
    pool.join()
    results = ()
    for i_tuple in range(len(individual_results[0])):
        results = results + (np.concatenate([_[i_tuple] for _ in individual_results]), )
    np.savez_compressed(f'{save_folder}/Voltr_subvolt', norm_subvolt=results[0], subvolt_=results[1])
    Path(save_folder+f'/finished_subvolt{fext}.tmp').touch()
    return None


if __name__ == "__main__":
    if len(sys.argv)>1:
        eval(sys.argv[1]+"()")
    else:
        dat_xls_file = pd.read_csv('Voltron_Log_DRN_Exp.csv', index_col=0)
        dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
        fext = ''
        for index, row in dat_xls_file.iterrows():
            folder = row['folder']
            fish = row['fish']
            task_type = row['task']
            if 'Social' in task_type[0]: # skip spike detection on social water task
                continue
            if row['subvolt']:
                continue
            save_folder = dat_folder + f'{folder}/{fish}/Data'
            save_image_folder = dat_folder + f'{folder}/{fish}/Results'
            if not os.path.isfile(save_folder+f'/finished_subvolt{fext}.tmp'):
                voltr2subvolt(row, fext=fext)
