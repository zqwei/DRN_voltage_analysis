import numpy as np
from pathlib import Path
import pandas as pd
from sys import platform
import os
from swim_dat import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
sns.set_style('ticks')


vol_file = Path('../Voltron_data/Voltron_Log_DRN_Exp.csv')
dat_xls_file = pd.read_csv(vol_file, index_col=0)
dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
# using Path to handle switches filesystems
if platform == "linux" or platform == "linux2":
    dir_folder = Path('/nrs/ahrens/Ziqiang/Takashi_DRN_project/ProcessedData/')
elif platform == 'win32':
    dir_folder = Path('U:\\Takashi') # put folder for windows system

    
import glob
if not os.path.isdir('swim_power'):
    os.mkdir('swim_power')
files = glob.glob('swim_power/*.png')
for f in files:
    os.remove(f)
plt.close('all')
valid_swim_list = []
for index, row in dat_xls_file.iterrows():
    valid_swim_list.append(valid_swim(row, sig_thres=0.5, isplot=True))
    plt.close('all')
    
swim_xls_file = dat_xls_file[valid_swim_list]

swim_xls_file.to_csv('depreciated/analysis_sections_based_on_swim_pattern.csv')
# swim_xls_file = pd.read_csv('analysis_sections_based_on_swim_pattern.csv', index_col=0)
# swim_xls_file['folder'] = swim_xls_file['folder'].apply(lambda x: f'{x:0>8}')