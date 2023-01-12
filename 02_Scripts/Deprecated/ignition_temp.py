# MCC Data Import and Pre-processing
#   by: Mark McKinnon and Conor McCoy
# ***************************** Run Notes ***************************** #
# - Prompts user for directory with MCC raw data                        #
#                                                                       #
# - Imports raw MCC data and creates excel sheets with header           #
#       information, raw data, and analyzed data (baseline and          #
#       mass loss corrected)                                            #
#                                                                       #
#                                                                       #
# TO DO:                                                                #
# - scan directory so that Excel sheets are not overwritten             #
#                                                                       #
# ********************************************************************* #

# --------------- #
# Import Packages #
# --------------- #
import os
import os.path # check for file existence
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import integrate
import git

label_size = 20
tick_size = 18
line_width = 2
legend_font = 10
fig_width = 10
fig_height = 6


def clean_file(file_name):
    fin = open(file_name, 'rt', encoding='UTF-16')
    fout = open(f'{file_name}_TEMP.tst', 'wt', encoding='UTF-16')
    # output file to write the result to
    for line in fin:
        # read replace the string and write to output file
        fout.write(line.replace('\t\t', '\t'))
    # close input and output files
    fin.close()
    fout.close()


def search_string_in_file(file_name, string_to_search):
    line_number = 0
    list_of_results = []
    with open(file_name, 'r', encoding='UTF-16') as read_obj:
        for line in read_obj:
            line_number += 1
            if string_to_search in line:
                line_num = line_number
    return line_num

def unique(list1):

    unique_list = []

    for x in list1:
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

data_dir = '../01_Data/'
save_dir = '../04_Computed/'

if not os.path.exists(save_dir): os.makedirs(save_dir)

# path = askdirectory(title='Select Folder') # shows dialog box and return the path -> this or a similar method can be used when interacting with database
# data_dir = path
# exp_names = []

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]
    if material == '.DS_Store':
        continue
    print(f'{material} MCC')
    ylims = [0, 0]
    xlims = [0, 0]
    data_df = pd.DataFrame()
    plot_data_df = pd.DataFrame()
    hoc_df = pd.DataFrame()
    ign_temp_df = pd.DataFrame()
    all_col_names = []
    if d.is_dir():
        if os.path.isdir(f'{d.path}/MCC'):
            for f in glob.iglob(f'{d.path}/MCC/*.txt'):
                if 'mass' in f.lower():
                    continue
                else:
                    # import data for each test
                    header_df = pd.read_csv(
                        f, header=None, sep='\t', nrows=3, index_col=0, squeeze=True)
                    initial_mass = float(header_df.at['Sample Weight (mg):'])
                    data_temp_df = pd.read_csv(
                        f, sep='\t', header=10, index_col='Time (s)')
                    fid = open(f.split('.txt')[0] + '_FINAL_MASS.txt', 'r')
                    final_mass = float(fid.readlines()[0].split('/n')[0])

                    col_name = f.split('.txt')[0].split('_')[-1]
                    # print(col_name)
                    if "O" not in col_name: # to ignore outliers (run code only for reptitions)
                        
                        all_col_names.append(col_name) # collect reptition numbers to account for botched tests (ex. R2, R3, R4, if R1 was bad)
                  
                        reduced_df = data_temp_df.loc[:, [
                        'Temperature (C)', 'HRR (W/g)']]
                        reduced_df[f'Time_copy_{col_name[-1]}'] = reduced_df.index  #col_name[-1] to have the reptition number as -1 (not -R1) to help Regex later

                        # Correct from initial mass basis to mass lost basis
                        reduced_df['HRR (W/g)'] = reduced_df['HRR (W/g)'] * \
                        (initial_mass / (initial_mass - final_mass))

                        max_lim = reduced_df['Temperature (C)'].iloc[-1] - (
                        (reduced_df['Temperature (C)'].iloc[-1]) % 50)
                        new_index = np.arange(120, int(max_lim) + 1)
                        new_data = np.empty((len(new_index),))
                        new_data[:] = np.nan
                        df_dict = {
                        'Temperature (C)': new_index, 'HRR (W/g)': new_data}
                        temp_df = pd.DataFrame(df_dict)

                       # Resample data to every temperature
                        reduced_df = pd.concat(
                        [reduced_df, temp_df], ignore_index=True)
                        reduced_df.set_index('Temperature (C)', inplace=True)
                        reduced_df.sort_index(inplace=True)
                        reduced_df.interpolate(
                        method='linear', axis=0, inplace=True)
                        reduced_df = reduced_df.loc[new_index, :]

                        reduced_df = reduced_df[~reduced_df.index.duplicated(
                        keep='first')]

                        # Baseline Correction
                        reduced_df['HRR correction'] = reduced_df.loc[120, 'HRR (W/g)'] + ((reduced_df.index - 120) / (
                        reduced_df.index.max() - 120)) * (reduced_df.loc[reduced_df.index.max(), 'HRR (W/g)'] - reduced_df.loc[120, 'HRR (W/g)'])
                        reduced_df[col_name] = reduced_df['HRR (W/g)'] - \
                        reduced_df['HRR correction']

                        data_df = pd.concat([data_df, reduced_df], axis=1)
                        data_array = data_df[col_name].to_numpy()
                        time_array = data_df[f'Time_copy_{col_name[-1]}'].to_numpy() #col_name[-1] to have the reptition number of the time column as -1 (not -R1) to help Regex later
                        data_array = data_array[~np.isnan(data_array)]
                        time_array = time_array[~np.isnan(time_array)]
                        hoc_df.at['Heat of Combustion (MJ/kg)', col_name] = (
                        integrate.trapz(y=data_array, x=time_array)) / 1000
                        hoc_df.at['Heat of Combustion (MJ/kg)', 'Mean'] = np.nan
                        hoc_df.at['Heat of Combustion (MJ/kg)', 'Std. Dev.'] = np.nan

            corrected_data = data_df.filter(regex='R[0-9]')  # TESTS WITHOUT Rnumber (ex. R1) are ignored and not used in HRR averaging or HoC determination.
            plot_data_df.loc[:, 'HRR_mean'] = corrected_data.mean(axis=1)
            plot_data_df.loc[:, 'HRR_std'] = corrected_data.std(axis=1)
            plot_data_df['HRR_upper'] = plot_data_df['HRR_mean'] + 2*plot_data_df['HRR_std']
            plot_data_df['HRR_lower'] = plot_data_df['HRR_mean'] - 2*plot_data_df['HRR_std']

        else:
            continue
    else:
        continue

    try:    
        upper_temp = min(plot_data_df.index[plot_data_df['HRR_upper']>33].tolist())
        lower_temp = min(plot_data_df.index[plot_data_df['HRR_lower']>17].tolist())
        mean_temp = min(plot_data_df.index[plot_data_df['HRR_mean']>25].tolist())
    except:
        upper_temp = np.nan
        lower_temp = np.nan
        mean_temp = np.nan

    ign_temp_df.loc['MCC','Lower (C)'] = lower_temp
    ign_temp_df.loc['MCC','Mean (C)'] = mean_temp
    ign_temp_df.loc['MCC','Upper (C)'] = upper_temp

    save_dir = f'../05_Computed/{material}/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ign_temp_df.to_csv(f'{save_dir}/ignition_temp.csv')

   # # trouble-shooting by outputting the dataframes
   #  if material == 'Coaxial_Cable' or  material == 'Pipe_Heat_Cable':
   #      data_df.to_csv(
   #      f'{data_dir}{material}/MCC/{material}_MCC_DataDF.csv', float_format='%.2f')
   #      corrected_data.to_csv(
   #      f'{data_dir}{material}/MCC/{material}_MCC_CorrectedDataDF.csv', float_format='%.2f')
   #      plot_data_df.to_csv(
   #      f'{data_dir}{material}/MCC/{material}_MCC_PlotDataDF.csv', float_format='%.2f')