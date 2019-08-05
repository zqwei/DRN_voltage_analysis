#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

import numpy as np
import pandas as pd
import os
from pathlib import Path
from preprocess_calcium import *
from utils_swim import *

vol_file = '../SnFR_data/SnFR_Log_DRN_Exp.csv'
dat_xls_file = pd.read_csv(vol_file)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/SnFRData/'
dir_folder = Path(dat_folder)


for index, row in dat_xls_file.iterrows():
    local_pca_demix(row)
