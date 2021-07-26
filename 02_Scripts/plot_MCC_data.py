# MCC Data Import and Pre-processing
#   by: Mark McKinnon
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
import glob
import numpy as np
import pandas as pd
import math
from tkinter import Tk
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import integrate


label_size = 20
tick_size = 18
line_width = 2
legend_font = 10
fig_width = 10
fig_height = 6

def clean_file(file_name):
    fin = open(file_name, 'rt', encoding = 'UTF-16')
    fout = open(f'{file_name}_TEMP.tst', 'wt', encoding = 'UTF-16')
    #output file to write the result to
    for line in fin:
        #read replace the string and write to output file
        fout.write(line.replace('\t\t', '\t'))
    #close input and output files
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

def create_1plot_fig():
    # Define figure for the plot
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    #plt.subplots_adjust(left=0.08, bottom=0.3, right=0.92, top=0.95)

    # Reset values for x & y limits
    x_min, x_max, y_min, y_max = 0, 0, 0, 0

    return(fig, ax1, x_min, x_max, y_min, y_max)

def plot_mean_data(df):
    ax1.plot(df.index, df.loc[:,'HRR_mean'], color='k', ls='-', marker=None, label = 'Mean Data')
    ax1.fill_between(df.index, df['HRR_mean'] - 2*df['HRR_std'], df['HRR_mean'] + 2*df['HRR_std'], color = 'k', alpha = 0.2)
    
    y_max = max(df.loc[:,'HRR_mean']+2*df.loc[:,'HRR_std'])
    y_min = min(df.loc[:,'HRR_mean']-2*df.loc[:,'HRR_std'])

    x_max = max(df.index)
    x_min = min(df.index)
    return(y_min, y_max, x_min, x_max)

def format_and_save_plot(xlims, ylims, file_loc):
    # Set tick parameters
    ax1.tick_params(labelsize=tick_size, length=8, width=0.75, direction = 'inout')

    # Scale axes limits & labels
    ax1.set_ylim(bottom=ylims[0], top=ylims[1])
    ax1.set_xlim(left=xlims[0], right=xlims[1])
    ax1.set_xlabel('Temperature (C)', fontsize=label_size)

    ax1.set_position([0.15, 0.3, 0.77, 0.65])

    y_range_array = np.arange(ylims[0], ylims[1] + 50, 50)
    ax1.set_ylabel('Specific HRR (W/g)', fontsize=label_size)    

    yticks_list = list(y_range_array)

    x_range_array = np.arange(xlims[0], xlims[1] + 50, 50)
    xticks_list = list(x_range_array)

    ax1.set_yticks(yticks_list)
    ax1.set_xticks(xticks_list)

    ax2 = ax1.secondary_yaxis('right')
    ax2.tick_params(axis='y', direction='in', length = 4)
    ax2.set_yticks(yticks_list)
    empty_labels = ['']*len(yticks_list)
    ax2.set_yticklabels(empty_labels)

    ax3 = ax1.secondary_xaxis('top')
    ax3.tick_params(axis='x', direction='in', length = 4)
    ax3.set_xticks(xticks_list)
    empty_labels = ['']*len(xticks_list)
    ax3.set_xticklabels(empty_labels)

    # Add legend
    handles1, labels1 = ax1.get_legend_handles_labels()

    # print(f'handles1: {handles1}')

    # # order = []
    # # order.append(labels1.index('Wet Sample Preparation'))
    # # order.append(labels1.index('Dry Sample Preparation'))

    # handles1 = handles1[i]
    # labels1 = labels1[i]

    # n_col, leg_list, leg_labels = legend_entries(handles1, labels1)

    #a_list = [a_list[i] for i in order]

    plt.legend(handles1, labels1, loc = 'upper center', bbox_to_anchor = (0.5, -0.23), fontsize=16,
                handlelength=2, frameon=True, framealpha=1.0, ncol=2)

    # Clean up whitespace padding
    #fig.tight_layout()

    # Save plot to file
    plt.savefig(file_loc)
    plt.close()

    print()

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

# path = askdirectory(title='Select Folder') # shows dialog box and return the path -> this or a similar method can be used when interacting with database
# data_dir = path
# exp_names = []

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]
    print(f'{material} MCC')
    ylims = [0,0]
    xlims = [0,0]
    fig, ax1, x_min, x_max, y_min, y_max = create_1plot_fig()
    data_df = pd.DataFrame()
    plot_data_df = pd.DataFrame()
    hoc_df = pd.DataFrame()
    if d.is_dir():
        if os.path.isdir(f'{d.path}/MCC'):
            for f in os.scandir(f'{d.path}/MCC/'):
                label_list = f.path.split('/')[-1].split('.')[0].split('_')
                if 'MASS' in label_list:
                    continue
                else:

                    # import data for each test
                    header_df = pd.read_csv(f, header = None, sep = '\t', nrows = 3, index_col = 0, squeeze = True)
                    initial_mass = float(header_df.at['Sample Weight (mg):'])
                    data_temp_df = pd.read_csv(f, sep = '\t', header = 10, index_col = 'Time (s)')
                    fid = open(f.path.split('.txt')[0] + '_FINAL_MASS.txt', 'r')
                    final_mass = float(fid.readlines()[0].split('/n')[0])

                    col_name = f.path.split('.txt')[0].split('_')[-1]

                    reduced_df = data_temp_df.loc[:,['Temperature (C)', 'HRR (W/g)']]
                    reduced_df[f'Time_copy_{col_name}'] = reduced_df.index

                    # Correct from initial mass basis to mass lost basis
                    reduced_df['HRR (W/g)'] = reduced_df['HRR (W/g)']*(initial_mass/(initial_mass-final_mass))

                    max_lim = reduced_df['Temperature (C)'].iloc[-1] - ((reduced_df['Temperature (C)'].iloc[-1])%50)
                    new_index = np.arange(150,int(max_lim)+1)
                    new_data = np.empty((len(new_index),))
                    new_data[:] = np.nan
                    df_dict = {'Temperature (C)': new_index, 'HRR (W/g)': new_data}
                    temp_df = pd.DataFrame(df_dict)

                    # Resample data to every temperature
                    reduced_df = pd.concat([reduced_df, temp_df], ignore_index = True)
                    reduced_df.set_index('Temperature (C)', inplace = True)
                    reduced_df.sort_index(inplace=True)
                    reduced_df.interpolate(method='linear', axis=0, inplace=True)
                    reduced_df = reduced_df.loc[new_index, :]

                    reduced_df = reduced_df[~reduced_df.index.duplicated(keep='first')]

                    # Baseline Correction
                    reduced_df['HRR correction'] = reduced_df.loc[150,'HRR (W/g)']+((reduced_df.index-150)/(reduced_df.index.max()-150))*(reduced_df.loc[reduced_df.index.max(),'HRR (W/g)'] - reduced_df.loc[150,'HRR (W/g)'])
                    reduced_df[col_name] = reduced_df['HRR (W/g)'] - reduced_df['HRR correction']

                    data_df = pd.concat([data_df, reduced_df], axis = 1)

                    data_array = data_df[col_name].to_numpy()
                    time_array = data_df[f'Time_copy_{col_name}'].to_numpy()
                    data_array = data_array[~np.isnan(data_array)]
                    time_array = time_array[~np.isnan(time_array)]

                    hoc_df.at['Heat of Combustion (MJ/kg)', col_name] = (integrate.trapz(y = data_array, x = time_array))/1000
                    hoc_df.at['Heat of Combustion (MJ/kg)', 'Mean'] = np.nan
                    hoc_df.at['Heat of Combustion (MJ/kg)', 'Std. Dev.'] = np.nan

            corrected_data = data_df.filter(regex = 'R[0-9]')
            plot_data_df.loc[:,'HRR_mean'] = corrected_data.mean(axis = 1)
            plot_data_df.loc[:,'HRR_std'] = corrected_data.std(axis = 1)

        else:
            continue
    else:
        continue

    mean_hoc = hoc_df.mean(axis = 1)
    std_hoc = hoc_df.std(axis = 1)

    hoc_df.at['Heat of Combustion (MJ/kg)', 'Mean'] = mean_hoc
    hoc_df.at['Heat of Combustion (MJ/kg)', 'Std. Dev.'] = std_hoc
    hoc_df = hoc_df[['R1', 'R2', 'R3', 'Mean', 'Std. Dev.']]

    ymin, ymax, xmin, xmax = plot_mean_data(plot_data_df)

    y_min = max(ymin, y_min)
    x_min = max(xmin, x_min)
    y_max = max(ymax, y_max)
    x_max = max(xmax, x_max)

    ylims[0] = 50 * (math.floor(y_min/50)-1)
    ylims[1] = 50 * (math.ceil(y_max/50)+1)
    xlims[0] = 50 * (math.floor(x_min/50))
    xlims[1] = 50 * (math.ceil(x_max/50))

    plot_dir = f'../03_Charts/{material}/MCC/'

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    format_and_save_plot(xlims, ylims, f'{plot_dir}{material}_MCC.pdf')

    hoc_df.to_csv(f'{plot_dir}{material}_MCC_Heats_of_Combustion.csv', float_format='%.2f')