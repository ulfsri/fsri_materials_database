# MCC Data Import and Pre-processing
#   by: Mark McKinnon and Craig Weinschenk
# ***************************** Run Notes ***************************** #
# - Prompts user for directory with MCC raw data                        #
#                                                                       #
# - Imports raw MCC data and creates excel sheets with header           #
#       information, raw data, and analyzed data (baseline and          #
#       mass loss corrected)                                            #
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
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

def plot_mean_data(df):
    y_upper = df['HRR_mean'] + 2*df['HRR_std']
    y_lower = df['HRR_mean'] - 2*df['HRR_std']

    fig.add_trace(go.Scatter(x=np.concatenate([df.index,df.index[::-1]]),y=pd.concat([y_upper,y_lower[::-1]]),
        fill='toself',hoveron='points',fillcolor='lightgray',line=dict(color='lightgray'),name='2'+ "\u03C3"))
    fig.add_trace(go.Scatter(x=df.index, y=df['HRR_mean'], marker=dict(color='black', size=8),name='Mean'))


    return()

def format_and_save_plot(file_loc):


    fig.update_layout(xaxis_title='Temperature (&deg;C)', font=dict(size=18))
    fig.update_layout(yaxis_title='Specific HRR (W/g)', title ='Specific HRR')
    fig.write_html(file_loc,include_plotlyjs="cdn")
    plt.close()
    print()

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]
    print(f'{material} MCC')
    ylims = [0,0]
    xlims = [0,0]
    fig = go.Figure()
    data_df = pd.DataFrame()
    plot_data_df = pd.DataFrame()
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

            corrected_data = data_df.filter(regex = 'R[0-9]')
            plot_data_df.loc[:,'HRR_mean'] = corrected_data.mean(axis = 1)
            plot_data_df.loc[:,'HRR_std'] = corrected_data.std(axis = 1)

        else:
            continue
    else:
        continue
    
    plot_mean_data(plot_data_df)

    plot_dir = f'../03_Charts/{material}/MCC/'

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    format_and_save_plot(f'{plot_dir}{material}_MCC.html')