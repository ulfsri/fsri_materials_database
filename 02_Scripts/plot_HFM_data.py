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
    columns_list = list(df.columns)
    colors = {'Wet':'b', 'Dry':'r'}
    for i in range(int(len(df.columns)/2)):
        col_ref = i-1
        mean_col = 2*(i-1)
        std_col = (2*i)-1
        data_lab = columns_list[col_ref].split('_')[0]
        c = colors[data_lab]
        ax1.plot(df.index, df.iloc[:,mean_col], color=c, ls='', marker='o', mew=1.0, mec='k', ms=5, label = data_lab)
        if df.iloc[:,std_col].isnull().all():
            y_max = max(df.iloc[:,mean_col])
            y_min = min(df.iloc[:,mean_col])
            continue
        else:
            ax1.errorbar(df.index, df.iloc[:,mean_col], yerr = 2*df.iloc[:,std_col], ls='', capsize = 5, elinewidth = 1.0, capthick = 1.0, ecolor='k')
            y_max = max(df.iloc[:,mean_col]+2*df.iloc[:,std_col])
            y_min = min(df.iloc[:,mean_col]-2*df.iloc[:,std_col])

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

    if file_loc.split('.')[0].split('_')[-1] == 'Conductivity':
        y_range_array = np.arange(ylims[0], ylims[1] + 0.05, 0.05)
        ax1.set_ylabel('Thermal Conductivity (W/mK)', fontsize=label_size)        
    else:
        y_range_array = np.arange(ylims[0], ylims[1] + 100, 100)
        ax1.set_ylabel('Specific Heat (J/kgK)', fontsize=label_size)    

    yticks_list = list(y_range_array)

    x_range_array = np.arange(xlims[0], xlims[1] + 5, 5)
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
    print(f'{material} Thermal Conductivity')
    ylims = [0,0]
    xlims = [0,0]
    fig, ax1, x_min, x_max, y_min, y_max = create_1plot_fig()
    k_wet = []
    k_dry = []
    if d.is_dir():
        for d_ in os.scandir(d):
            if d_.is_dir() and 'HFM' in d_.path:
                for f in os.scandir(d_):
                    if 'Conductivity' in f.path:
                        if 'Dry' in f.path:
                            k_dry.append(f.path)
                        else:
                            k_wet.append(f.path)
                    else:
                        continue
        dirs = [k_wet, k_dry]
        k_plot_data = pd.DataFrame()
        for coll in dirs:
            k_df = pd.DataFrame()

            for file_path in coll:
                if 'Conductivity' in file_path:
                    clean_file(file_path)
                    start_line = search_string_in_file(f'{file_path}_TEMP.tst', 'Results Table -- SI Units')
                    header_line = start_line+1
                    temp_df = pd.read_csv(f'{file_path}_TEMP.tst', sep = '\t', header = header_line, usecols = [1,2,3,4], index_col = 'Mean Temp', engine='python', skip_blank_lines = False, encoding = 'UTF-16')
                    os.remove(f'{file_path}_TEMP.tst')
                    temp_df.dropna(axis=0, inplace = True)

                    temp_df_ind = temp_df.index.to_series()
                    temp_df.index = temp_df_ind.round(1)

                    f_str = file_path.split('/')[-1].split('_')
                    rep = f_str[-1].split('.')[0]

                    col = f'{f_str[-4]}_{rep}'

                    k_df[col] = temp_df['Average Cond']

            index_list = k_df.index.to_list()
            unique_indices = unique(index_list)

            for i in unique_indices:
                data_df = k_df.loc[i,:]
                if isinstance(data_df, pd.DataFrame):
                    data_df = data_df.stack()
                i_mean = data_df.mean()
                i_std = data_df.std()
                
                k_plot_data.at[i,f'{f_str[-4]}_mean'] = i_mean
                k_plot_data.at[i,f'{f_str[-4]}_std'] = i_std

        ymin, ymax, xmin, xmax = plot_mean_data(k_plot_data)

        y_min = max(ymin, y_min)
        x_min = max(xmin, x_min)
        y_max = max(ymax, y_max)
        x_max = max(xmax, x_max)

        ylims[0] = 0.05 * (math.floor(y_min/0.05)-1)
        ylims[1] = 0.05 * (math.ceil(y_max/0.05)+1)
        xlims[0] = 5 * (math.floor(x_min/5)-1)
        xlims[1] = 5 * (math.ceil(x_max/5)+1)

        plot_dir = f'../03_Charts/{material}/HFM/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        format_and_save_plot(xlims, ylims, f'{plot_dir}{material}_Thermal_Conductivity.pdf')

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]
   
    temp_df = pd.DataFrame() 
    density_df = pd.DataFrame(index = ['mean', 'std'])
    if d.is_dir():
        for d_ in os.scandir(d):
            if d_.is_dir() and 'Density' in d_.path:
                for f in os.scandir(d_):
                    temp_density_series = pd.read_csv(f, squeeze = True, index_col = 0)
                    f_str = f.path.split('.')[-2].split('_')[-3] + '_' + f.path.split('.')[-2].split('_')[-1]
                    temp_df.at['Density', f_str] = temp_density_series['Density']

    if temp_df.empty:
        continue
    else:
        wet_mean_density = temp_df.filter(regex = 'Wet').mean(axis = 1).at['Density']
        wet_std_density = temp_df.filter(regex = 'Wet').std(axis = 1).at['Density']
        dry_mean_density = temp_df.filter(regex = 'Dry').mean(axis = 1).at['Density']
        dry_std_density = temp_df.filter(regex = 'Dry').std(axis = 1).at['Density']

    data = {'mean': [wet_mean_density, dry_mean_density], 'std': [wet_std_density, dry_std_density]}
    density_df = pd.DataFrame.from_dict(data, orient='index', columns = ['Wet', 'Dry'])

    print(f'{material} Heat Capacity')
    ylims = [0,0]
    xlims = [0,0]
    fig, ax1, x_min, x_max, y_min, y_max = create_1plot_fig()
    c_wet = []
    c_dry = []
    if d.is_dir():
        file_counter = 0
        for d_ in os.scandir(d):
            if d_.is_dir() and 'HFM' in d_.path:
                for f in os.scandir(d_):
                    if 'HeatCapacity' in f.path:
                        file_counter += 1
                        if 'Dry' in f.path:
                            c_dry.append(f.path)
                        else:
                            c_wet.append(f.path)

        if file_counter == 0:
            print('empty')
            continue
        dirs = [c_wet, c_dry]
        c_plot_data = pd.DataFrame()
        for coll in dirs:
            c_df = pd.DataFrame()
            for file_path in coll:
                if 'HeatCapacity' in file_path:
                    clean_file(file_path)
                    start_line = search_string_in_file(f'{file_path}_TEMP.tst', 'Results Table -- SI Units')
                    header_line = start_line+1
                    temp_df = pd.read_csv(f'{file_path}_TEMP.tst', sep = '\t', header = header_line, skiprows = [header_line+1], engine='python', skip_blank_lines = False, encoding = 'UTF-16')
                    os.remove(f'{file_path}_TEMP.tst')
                    temp_df.dropna(axis=1, inplace = True)
                    temp_df.rename(columns={'Unnamed: 0':'Mean Temp', 'Mean Temp':'Enthalpy', 'Enthalpy   ':'Specific Heat'}, inplace = True)
                    temp_df.set_index('Mean Temp', inplace = True)

                    temp_df_ind = temp_df.index.to_series()
                    temp_df.index = temp_df_ind.round(1)

                    f_str = file_path.split('/')[-1].split('_')
                    rep = f_str[-1].split('.')[0]

                    col = f'{f_str[-4]}_{rep}'

                    c_df[col] = temp_df['Specific Heat']

            index_list = c_df.index.to_list()
            unique_indices = unique(index_list)

            for i in unique_indices:
                data_df = c_df.loc[i,:]
                if isinstance(data_df, pd.DataFrame):
                    data_df = data_df.stack()
                i_mean = data_df.mean()
                i_std = data_df.std()
                
                c_plot_data.at[i,f'{f_str[-4]}_mean'] = i_mean / density_df.at['mean', f_str[-4]]
                c_plot_data.at[i,f'{f_str[-4]}_std'] = i_std / density_df.at['mean', f_str[-4]]

        ymin, ymax, xmin, xmax = plot_mean_data(c_plot_data)

        y_min = max(ymin, y_min)
        x_min = max(xmin, x_min)
        y_max = max(ymax, y_max)
        x_max = max(xmax, x_max)

        ylims[0] = 100 * (math.floor(y_min/100)-1)
        ylims[1] = 100 * (math.ceil(y_max/100)+1)
        xlims[0] = 5 * (math.floor(x_min/5)-1)
        xlims[1] = 5 * (math.ceil(x_max/5)+1)

        plot_dir = f'../03_Charts/{material}/HFM/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        print('Plotting Chart')

        format_and_save_plot(xlims, ylims, f'{plot_dir}{material}_Specific_Heat.pdf')

#     f_name = file.split('.txt')[0]
#     sample_name = f_name.split('_')[0]      

#     header_df = pd.read_csv(f'{path}/{file}', header = 0, sep = '\t', nrows = 7, index_col = 'Sample ID:')
#     raw_df = pd.read_csv(f'{path}/{file}', header = 11, sep = '\t', index_col = 'Time (s)')

#     reduced_df = raw_df.loc[:,['Temperature (C)','HRR (W/g)']]

#     # Correct from initial mass basis to mass lost basis
#     reduced_df['HRR (W/g)'] = reduced_df['HRR (W/g)']*(float(header_df.loc['Sample Weight (mg):'])/(float(header_df.loc['Sample Weight (mg):'])-float(header_df.loc['Final Weight (mg):'])))
#     max_lim = reduced_df['Temperature (C)'].iloc[-1] - ((reduced_df['Temperature (C)'].iloc[-1])%50)
#     new_index = np.arange(150,int(max_lim)+1)
#     new_data = np.empty((len(new_index),))
#     new_data[:] = np.nan
#     df_dict = {'Temperature (C)': new_index, 'HRR (W/g)': new_data}
#     temp_df = pd.DataFrame(df_dict)

#     # Resample data to every temperature
#     reduced_df = pd.concat([reduced_df, temp_df], ignore_index = True)
#     reduced_df.set_index('Temperature (C)', inplace = True)
#     reduced_df.sort_index(inplace=True)
#     reduced_df.interpolate(method='linear', axis=0, inplace=True)
#     reduced_df = reduced_df.loc[new_index, :]

#     # Baseline Correction
#     reduced_df['HRR correction'] = reduced_df.loc[150,'HRR (W/g)']+((reduced_df.index-150)/(reduced_df.index.max()-150))*(reduced_df.loc[reduced_df.index.max(),'HRR (W/g)'] - reduced_df.loc[150,'HRR (W/g)'])
#     reduced_df['HRR (W/g)'] = reduced_df['HRR (W/g)'] - reduced_df['HRR correction']
#     reduced_df.drop(columns = 'HRR correction', inplace = True)

#     d[f'{f_name}_header'] = header_df
#     d[f'{f_name}_raw'] = raw_df
#     d[f'{f_name}_reduced'] = reduced_df

# status = pd.read_csv('../02_Info/Materials_Status.csv', header = 0, index_col = 'Material', usecols = ['Material','MCC'])
# materials_list = list(status[status['MCC'] == 'C'].index)
# materials_list = [mat.replace(' ', '_') for mat in materials_list]

# # create a mean data worksheet and calculate the mean HRR and THR
# for i in materials_list:
#     d[f'{i}_30_mean'] = pd.DataFrame()
#     for key in d.keys():   
#         if i in key and key.split('_')[-1] == 'reduced':
#             d[f'{i}_30_mean'] = pd.concat([d[f'{i}_30_mean'], d[key]], axis = 0)
#     d[f'{i}_30_mean'].reset_index(inplace = True)
#     d[f'{i}_30_mean'].loc[:,'Temperature (C)'] = d[f'{i}_30_mean'].loc[:,'Temperature (C)'].astype('int32')
#     d[f'{i}_30_mean'] = d[f'{i}_30_mean'].groupby(by = ['Temperature (C)'], as_index=True, sort = True).mean()
#     d[f'{i}_30_mean'].loc[:,'Total Heat Released (J/g)'] = d[f'{i}_30_mean']['HRR (W/g)'].cumsum()

# # Save an excel worksheet for each material
# for i in materials_list:
#     sheet_names = []
#     for key in d.keys():   
#         if i in key:
#             sheet_names.append(key)

#     with pd.ExcelWriter(f'../01_Data/MCC/{i}.xlsx') as writer:
#         for sheet in sheet_names:
#             if sheet.split('_')[-1] == 'header':
#                 d[sheet].to_excel(writer, sheet_name = sheet, header = False)
#             else:
#                 d[sheet].to_excel(writer, sheet_name = sheet)
