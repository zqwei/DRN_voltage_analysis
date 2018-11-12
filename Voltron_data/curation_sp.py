import warnings
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
from fish_proc.utils.memory import get_process_memory, clear_variables
import pandas as pd
from DRN_processing_jobs import demix_middle_data_with_mask
from DRN_voltr_spikes import voltron, voltr2spike, voltr2subvolt
sns.set(font_scale=2)
sns.set_style("white")
from sys import platform

if platform=='win32':
    dat_folder = r'Z://Takashi_DRN_project//'
else:
    dat_folder = '/nrs/ahrens/Ziqiang/Takashi_DRN_project/'

dat_xls_file = pd.read_csv(dat_folder + 'Voltron Log_DRN_Exp.csv', index_col=0)
if 'index' in dat_xls_file.columns:
    dat_xls_file = dat_xls_file.drop('index', axis=1)
dat_xls_file['folder'] = dat_xls_file['folder'].astype(int).apply(str)

curate_dat = dat_xls_file[dat_xls_file['FineTune'] & dat_xls_file['dataToAnalysis']]


if __name__ == "__main__":
    if len(sys.argv)==1:
        for index, row in curate_dat.iterrows():
            demix_middle_data_with_mask(row)
            voltron(row, fext='_with_mask', is_mask=True)
            voltr2spike(row, fext='_with_mask')
            voltr2subvolt(row, fext='_with_mask')
    else:
        row = curate_dat.iloc[int(sys.argv[1])]
        demix_middle_data_with_mask(row)
        voltron(row, fext='_with_mask', is_mask=True)
        voltr2spike(row, fext='_with_mask')
        voltr2subvolt(row, fext='_with_mask')
        