#!/groups/ahrens/home/weiz/miniconda3/envs/myenv/bin/python

from voltr_spike import *
from voltr_ablation import *

ablt_len = 6
ablt_sovo_len = 26

if __name__ == "__main__":
    if len(sys.argv)>1:
        eval(sys.argv[1]+"()")
    else:
        dat_xls_file = pd.read_csv('Voltron_Log_DRN_Exp.csv', index_col=0)
        dat_xls_file['folder'] = dat_xls_file['folder'].apply(lambda x: f'{x:0>8}')
        fext = ''

        # first ablation data
        ablt_sovo_str_alt_set = ['', '-swimonly_visualonly']
        ablt_len_set = [ablt_len, ablt_sovo_len]
        for ablt_sovo_str_alt, ablt_len_ in zip(ablt_sovo_str_alt_set, ablt_len_set):
            for index, row in dat_xls_file.iterrows():
                ablation_pair = search_paired_data(row, dat_xls_file, ablt_len=ablt_len_, ablt_sovo_str_alt=ablt_sovo_str_alt)
                if not ablation_pair:
                    continue
                folder = row['folder']
                fish = row['fish']
                task_type = row['task']
                save_folder = dat_folder + f'{folder}/{fish}/Data'
                if not os.path.exists(save_folder):
                    continue
                if not os.path.isfile(save_folder+f'/finished_voltr{fext}.tmp'):
                    x, y = align_components(row, ablt_len=ablt_len_, ablt_sovo_str_alt=ablt_sovo_str_alt)
                    voltron_ablt(row, x, y, fext='', is_mask=True, ablt_len=ablt_len_, ablt_sovo_str_alt=ablt_sovo_str_alt)

        # other
        for index, row in dat_xls_file.iterrows():
            folder = row['folder']
            fish = row['fish']
            task_type = row['task']
            if 'Social' in task_type[0]: # skip spike detection on social water task
                continue
            if row['voltr']:
                continue
            save_folder = dat_folder + f'{folder}/{fish}/Data'
            save_image_folder = dat_folder + f'{folder}/{fish}/Results'
            if not os.path.isfile(save_folder+f'/finished_voltr{fext}.tmp'):
                voltron(row, fext=fext, is_mask=True)

        # spikes
        for index, row in dat_xls_file.iterrows():
            folder = row['folder']
            fish = row['fish']
            task_type = row['task']
            save_folder = dat_folder + f'{folder}/{fish}/Data'
            if row['spikes']:
                continue
            if not os.path.isfile(save_folder+f'/finished_spikes{fext}.tmp'):
                print([folder, fish, task_type])
                if 'dendrite' in fish:
                    win_=10001
                else:
                    win_=50001
                voltr2spike(row, fext=fext, cpu=True, win_=win_)
