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
import subprocess

label_size = 20
tick_size = 18
line_width = 2
legend_font = 10
fig_width = 10
fig_height = 6

def create_1plot_fig():
    # Define figure for the plot
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    #plt.subplots_adjust(left=0.08, bottom=0.3, right=0.92, top=0.95)

    # Reset values for x & y limits
    x_min, x_max, y_min, y_max = 0, 0, 0, 0

    return(fig, ax1, x_min, x_max, y_min, y_max)

def plot_mean_data(df):
    color_dict = {'MEAS':'k', 'REF': 'r','BL': 'b'}
    label_dict = {'MEAS':'Mean Sample Measurement', 'REF': 'Reference Measurement','BL': 'Baseline Measurement'}
    for i in ['MEAS', 'REF', 'BL']:
        ax1.plot(df.index, df[f'{i}_mean'], color=color_dict[i], ls='-', marker=None, lw=1.0, label = label_dict[i])
        try:
            ax1.fill_between(df.index, df[f'{i}_mean'] - 2*df[f'{i}_std'], df[f'{i}_mean'] + 2*df[f'{i}_std'], color =color_dict[i], alpha = 0.2)
        except:
            continue
        y_max = max(df.loc[:,'REF_mean']+2*df.loc[:,'REF_std'])
        y_min = min(df.loc[:,'BL_mean']-2*df.loc[:,'BL_std'])

    x_max = max(df.index)
    x_min = min(df.index)
    return(y_min, y_max, x_min, x_max)

def format_and_save_plot(xlims, ylims, file_loc):
    # Set tick parameters
    ax1.tick_params(labelsize=tick_size, length=8, width=0.75, direction = 'inout')

    # Scale axes limits & labels
    ax1.set_ylim(bottom=ylims[0], top=ylims[1])
    ax1.set_xlim(left=xlims[0], right=xlims[1])
    ax1.set_xlabel('Wavelength (nm)', fontsize=label_size)

    ax1.set_position([0.15, 0.3, 0.77, 0.65])

    y_range_array = np.arange(ylims[0], ylims[1] + 0.2, 0.2)
    ax1.set_ylabel('Reflection Signal (-)', fontsize=label_size)        

    yticks_list = list(y_range_array)

    x_range_array = np.arange(xlims[0], xlims[1] + 1000, 1000)
    xticks_list = list(x_range_array)

    ax1.set_yticks(yticks_list)
    ax1.set_xticks(xticks_list)
    ax1.set_xticklabels(xticks_list, rotation = 40, ha='right')

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

    #Get github hash to display on graph
    label = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

    ax1.text(1, 1,'Repository Version: ' + label,
         horizontalalignment='right',
         verticalalignment='bottom',
         transform = ax1.transAxes)

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

    plt.legend(handles1, labels1, loc = 'right', fontsize=12,
               handlelength=2, frameon=True, framealpha=0.5, ncol=1)

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
    data = pd.DataFrame()
    data_df = pd.DataFrame()
    material = d.path.split('/')[-1]
    print(f'{material} IS')
    ylims = [0,0]
    xlims = [0,0]
    fig, ax1, x_min, x_max, y_min, y_max = create_1plot_fig()
    if d.is_dir():
        if os.path.isdir(f'{d.path}/FTIR'):
            for f in os.scandir(f'{d.path}/FTIR/IS/'):
                label_list = f.path.split('/')[-1].split('.')[0].split('_')
                col_name = f'{label_list[-4]}_{label_list[-3]}_{label_list[-1]}'
                temp_df = pd.read_csv(f, header = None) # index_col = 0

                temp_df.rename(columns = {0:'wavenumber', 1: col_name}, inplace=True)
                temp_df['wavelength'] = 10000000/temp_df.iloc[:,0] # wavelength in nm

                data = pd.concat([data, temp_df], axis = 1)               
        else:
            continue

        reflect_data = data.filter(regex = 'REFLECT')
        bl_data = reflect_data.filter(regex = '_BL_') 
        ref_data = reflect_data.filter(regex = '_REF_')
        meas_data = reflect_data.filter(regex = '_MEAS_')

        data_df.loc[:,'BL_mean'] = bl_data.mean(axis = 1)
        data_df.loc[:,'BL_std'] = bl_data.std(axis = 1)
        data_df.loc[:,'REF_mean'] = ref_data.mean(axis = 1)
        data_df.loc[:,'REF_std'] = ref_data.std(axis = 1)
        data_df.loc[:,'MEAS_mean'] = meas_data.mean(axis = 1)
        data_df.loc[:,'MEAS_std'] = meas_data.std(axis = 1)

        data.dropna(axis = 0, how = 'any', inplace = True)

        data_df.loc[:,'wavelength'] = data.groupby(by=data.columns, axis=1).mean().loc[:,'wavelength']
        data_df.set_index('wavelength', inplace=True)

        ymin, ymax, xmin, xmax = plot_mean_data(data_df)

        y_min = max(ymin, y_min)
        x_min = max(xmin, x_min)
        y_max = max(ymax, y_max)
        x_max = max(xmax, x_max)

        # ylims[0] = 0.05 * (math.floor(y_min/0.05)-1)
        # ylims[1] = 0.05 * (math.ceil(y_max/0.05)+1)
        ylims[0] = 0
        ylims[1] = 1.05
        xlims[0] = 1000 * math.floor(x_min/1000)
        xlims[1] = 1000 * math.ceil(x_max/1000)

        plot_dir = f'../03_Charts/{material}/FTIR/IS/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        format_and_save_plot(xlims, ylims, f'{plot_dir}{material}_IS_Reflection.pdf')
