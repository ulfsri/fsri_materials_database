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


label_size = 20
tick_size = 18
line_width = 2
legend_font = 10
fig_width = 10
fig_height = 6

def apply_savgol_filter(raw_data):
    raw_data = raw_data.dropna().loc[0:]
    converted_data = savgol_filter(raw_data,15,3)
    filtered_data = pd.Series(converted_data, index=raw_data.index.values)
    return(filtered_data.loc[0:])

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

    hr_dict = {'3K_min':'r', '10K_min':'g', '30K_min':'b'}

    for i in hr_dict.keys():
        mean_df = df.filter(regex = 'mean')
        mean_hr_df_temp = mean_df.filter(regex = i)
        mean_hr_df = mean_hr_df_temp.dropna(axis = 'index')
        std_df = df.filter(regex = 'std')
        std_hr_df_temp = std_df.filter(regex = i)
        std_hr_df = std_hr_df_temp.dropna(axis = 'index')

        upper_lim = mean_hr_df.iloc[:,0] + 2*std_hr_df.iloc[:,0]
        lower_lim = mean_hr_df.iloc[:,0] - 2*std_hr_df.iloc[:,0]

        i_str = i.replace('_','/')

        ax1.plot(mean_hr_df.index, mean_hr_df, color=hr_dict[i], ls='-', marker=None, label = i_str)
        ax1.fill_between(upper_lim.index, lower_lim, upper_lim, color = hr_dict[i], alpha = 0.2)

    y_max = upper_lim.max()
    y_min = lower_lim.min()

    x_max = max(df.index)
    x_min = min(df.index)
    return(y_min, y_max, x_min, x_max)

def format_and_save_plot(xlims, ylims, inc, file_loc):
    axis_dict = {'Mass': 'Normalized Mass', 'MLR': 'Normalized MLR (1/s)', 'Flow': 'Heat Flow Rate (W/g)'}
    keyword = file_loc.split('.pdf')[0].split('_')[-1]

    # Set tick parameters
    ax1.tick_params(labelsize=tick_size, length=8, width=0.75, direction = 'inout')

    # Scale axes limits & labels
    ax1.set_ylim(bottom=ylims[0], top=ylims[1])
    ax1.set_xlim(left=xlims[0], right=xlims[1])
    ax1.set_xlabel('Temperature (C)', fontsize=label_size)

    ax1.set_position([0.15, 0.3, 0.77, 0.65])

    y_range_array = np.arange(ylims[0], ylims[1] + inc, inc)
    ax1.set_ylabel(axis_dict[keyword], fontsize=label_size)    

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
                handlelength=2, frameon=True, framealpha=1.0, ncol=3)

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

plot_dict = {'Normalized Mass':'Mass', 'Normalized MLR':'MLR', 'Heat Flow Rate':'Heat_Flow'}

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]
    plot_data_df = pd.DataFrame()
    print(f'{material} STA')
    if d.is_dir():
        if os.path.isdir(f'{d.path}/STA/'):
            for d_ in os.scandir(f'{d.path}/STA/N2/'):
                data_df = pd.DataFrame()
                reduced_df = pd.DataFrame()
                for f in os.scandir(d_.path):
                    print(f.path)
                    HR = d_.path.split('/')[-1]
                    if 'Meta' in f.path:
                        continue
                    else:

                        # import data for each test
                        data_temp_df = pd.read_csv(f, header = 0)
                        data_temp_df.rename(columns = {'##Temp./°C':'Temp (C)', 'Time/min':'time (s)'}, inplace = True)
                        data_temp_df['Mass/%'] = data_temp_df['Mass/%']/data_temp_df.loc[0,'Mass/%']
                        data_temp_df['time (s)'] = (data_temp_df['time (s)']-data_temp_df.loc[0,'time (s)'])*60
                        data_temp_df['Normalized MLR (1/s)'] = -data_temp_df['Mass/%'].diff()/data_temp_df['time (s)'].diff()
                        data_temp_df['Normalized MLR (1/s)'] = apply_savgol_filter(data_temp_df['Normalized MLR (1/s)'])

                        col_name = f.path.split('.csv')[0].split('_')[-1]

                        min_lim = data_temp_df['Temp (C)'].iloc[1] - ((data_temp_df['Temp (C)'].iloc[1])%1)
                        max_lim = data_temp_df['Temp (C)'].iloc[-1] - ((data_temp_df['Temp (C)'].iloc[-1])%1)

                        reduced_df = data_temp_df.loc[:,['Temp (C)', 'Mass/%', 'Normalized MLR (1/s)', 'DSC/(mW/mg)']]

                        new_index = np.arange(int(min_lim),int(max_lim)+1)
                        new_data = np.empty((len(new_index),))
                        new_data[:] = np.nan
                        df_dict = {'Temp (C)': new_index, 'Normalized Mass': new_data, 'Normalized MLR (1/s)': new_data, 'Heat Flow Rate (W/g)': new_data}
                        temp_df = pd.DataFrame(df_dict)

                        # Resample data to every temperature
                        reduced_df = pd.concat([reduced_df, temp_df], ignore_index = True)
                        reduced_df.set_index('Temp (C)', inplace = True)
                        reduced_df.sort_index(inplace=True)
                        reduced_df.interpolate(method='linear', axis=0, inplace=True)
                        reduced_df = reduced_df.loc[new_index, :]

                        reduced_df['Normalized Mass'] = reduced_df['Mass/%']
                        reduced_df['Heat Flow Rate (W/g)'] = reduced_df['DSC/(mW/mg)']
                        reduced_df.drop(labels = ['Mass/%', 'DSC/(mW/mg)'], axis = 1, inplace = True)

                        reduced_df = reduced_df[~reduced_df.index.duplicated(keep='first')]

                        if data_df.empty:
                            data_df = reduced_df
                        else:
                            data_df = pd.concat([data_df, reduced_df], axis = 1)

                for m in plot_dict.keys():
                    data_sub = data_df.filter(regex = m)
                    plot_data_df.loc[:,f'{m} {HR} mean'] = data_df.filter(regex = m).mean(axis = 1)
                    plot_data_df.loc[:,f'{m} {HR} std'] = data_df.filter(regex = m).std(axis = 1)

        else:
            continue
    else:
        continue

    # plot_data_df.to_csv(f'{data_dir}{material}/STA/N2/TEST.csv')

    plot_dir = f'../03_Charts/{material}/STA/N2/'

    plot_inc = {'Mass': 0.2, 'MLR': 0.001, 'Heat_Flow': 0.5}

    for m in plot_dict.keys():    
        ylims = [0,0]
        xlims = [0,0]
        fig, ax1, x_min, x_max, y_min, y_max = create_1plot_fig()

        plot_data = plot_data_df.filter(regex = m)
        ymin, ymax, xmin, xmax = plot_mean_data(plot_data)

        y_min = max(ymin, y_min)
        x_min = max(xmin, x_min)
        y_max = max(ymax, y_max)
        x_max = max(xmax, x_max)

        inc = plot_inc[plot_dict[m]]

        ylims[0] = inc * (math.floor(y_min/inc))
        ylims[1] = inc * (math.ceil(y_max/inc))
        xlims[0] = 50 * (math.floor(x_min/50))
        xlims[1] = 50 * (math.ceil(x_max/50))

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        suffix = plot_dict[m]
        format_and_save_plot(xlims, ylims, inc, f'{plot_dir}{material}_STA_{suffix}.pdf')