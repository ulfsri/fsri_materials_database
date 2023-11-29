# Micro-scale calorimeter html data processing script
#   by: ULRI's Fire Safety Research Institute
#   Questions? Submit them here: https://github.com/ulfsri/fsri_materials_database/issues

# ***************************** Usage Notes *************************** #
# - Script outputs as a function of temperature                         #
#   -  HTML Graphs dir: /03_Charts/{Material}/MCC                       #
#      Graphs: Specific HRR                                             #
#                                                                       #
#      HTML Tables dir: /01_Data/{Material}/MCC                         #
#      Tables: Heat of Combustion                                       #
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
import git
from scipy import integrate
from pybaselines import Baseline, utils

plot_all = True
if not plot_all: 
    print('plot_all is set to False, so any materials with existing html output files will be skipped')

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

def reindex_data(data, min_x, max_x, dx):
    data = data.loc[~data.index.duplicated(), :]
    data = data.reindex(data.index.union(np.arange(min_x, max_x+dx, dx)))
    data = data.astype('float64')
    data.index = data.index.astype('float64')
    data = data.interpolate(method='cubic')

    data = data.loc[np.arange(min_x, max_x+dx, dx)]

    return data

def format_and_save_plot(file_loc):

    fig.update_layout(xaxis_title='Temperature (&deg;C)', font=dict(size=18))
    fig.update_layout(yaxis_title='Specific HRR (W/g)')
    # fig.update_layout(autosize=False, width=513, height=450,margin=dict(l=25,r=25,b=40,t=40,pad=4))
    fig.update_layout(margin=dict(l=25,r=25,b=40,t=40,pad=4))

    #Get github hash to display on graph
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.commit.hexsha
    short_sha = repo.git.rev_parse(sha, short=True)

    fig.add_annotation(dict(font=dict(color='black',size=15),
                                        x=1,
                                        y=1.02,
                                        showarrow=False,
                                        text="Repository Version: " + short_sha,
                                        textangle=0,
                                        xanchor='right',
                                        xref="paper",
                                        yref="paper"))

    fig.write_html(file_loc,include_plotlyjs="cdn")
    plt.close()
    # print()

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

# initialize material status dataframe
if os.path.isfile('Utilities/material_status.csv'):
    mat_status_df = pd.read_csv('Utilities/material_status.csv', index_col = 'material')
else:
    mat_status_df = pd.DataFrame(columns = ['Wet_cp', 'Dry_cp', 'Wet_k', 'Dry_k', 'STA_MLR', 'CONE_MLR_25', 'CONE_MLR_50', 'CONE_MLR_75', 'CONE_HRRPUA_25', 'CONE_HRRPUA_50', 'CONE_HRRPUA_75', 'CO_Yield', 'MCC_HRR', 'Soot_Yield', 'MCC_HoC', 'Cone_HoC', 'HoR', 'HoG', 'MCC_Ign_Temp', 'Melting_Temp', 'Emissivity', 'Full_JSON', "Picture"])

    for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
        if os.path.isdir(f'{data_dir}/{d}'):
            material = d

            r = np.empty((23, ))
            r[:] = np.nan
            mat_status_df.loc[material, :] = r
    mat_status_df.fillna(False, inplace=True)

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".") and f != 'README.md'), key=str.lower):
    material = d

    if not plot_all:
        output_exists = False
        for c in ['MCC_HRR', 'MCC_HoC', 'MCC_Ign_Temp']: 
            if mat_status_df.loc[material, c]: output_exists = True
        if output_exists: 
            # print(f'Skipping {material} MCC --- plot_all is False and output charts exist')
            continue

    if os.path.isdir(f'{data_dir}{d}/MCC/'):
        fig = go.Figure()
        data_df = pd.DataFrame()
        plot_data_df = pd.DataFrame()
        hoc_df_html = pd.DataFrame()
        all_col_names = []
        print(material + ' MCC')
        for f in glob.iglob(f'{data_dir}{d}/MCC/*.txt'):
            if 'mass' in f.lower():
                continue
            else:
                # import data for each test
                header_df = pd.read_csv(f, header = None, sep = '\t', nrows = 3, index_col = 0).squeeze()
                initial_mass = float(header_df.at['Sample Weight (mg):'])
                data_temp_df = pd.read_csv(f, sep = '\t', header = 10, index_col = 'Time (s)')
                fid = open(f.split('.txt')[0] + '_FINAL_MASS.txt', 'r')
                final_mass = float(fid.readlines()[0].split('/n')[0])
                col_name = f.split('.txt')[0].split('_')[-1]

                if "O" not in col_name: # to ignore outliers (run code only for repetitions)

                    all_col_names.append(col_name) #
                    reduced_df = data_temp_df.loc[:,['Temperature (C)', 'HRR (W/g)']]
                    reduced_df[f'time_{col_name}'] = reduced_df.index 

                    # Correct from initial mass basis to mass lost basis
                    reduced_df['HRR (W/g)'] = reduced_df['HRR (W/g)']*(initial_mass/(initial_mass-final_mass))

                    max_lim = reduced_df['Temperature (C)'].iloc[-1] - ((reduced_df['Temperature (C)'].iloc[-1])%50)
                    # new_index = np.arange(150,int(max_lim)+1)

                    reduced_df.set_index('Temperature (C)', inplace=True)
                    reduced_df = reindex_data(reduced_df, 150, max_lim, 1)

                    # Baseline Correction
                    reduced_df['HRR correction'] = reduced_df.loc[150,'HRR (W/g)']+((reduced_df.index-150)/(reduced_df.index.max()-150))*(reduced_df.loc[reduced_df.index.max(),'HRR (W/g)'] - reduced_df.loc[150,'HRR (W/g)'])
                    reduced_df[col_name] = reduced_df['HRR (W/g)'] - reduced_df['HRR correction']

                    data_array = reduced_df[col_name].to_numpy()
                    time_array = reduced_df[f'time_{col_name}'].to_numpy()
                    data_array = data_array[~np.isnan(data_array)]
                    time_array = time_array[~np.isnan(time_array)]
                    time_array = np.unique(time_array)

                    hoc_df_html.at['Heat of Combustion (MJ/kg)', col_name] = (integrate.trapz(y=data_array, x=time_array)) / 1000

                    # Alternative Baseline Correction (NOT CURRENTLY IMPLEMENTED)
                    x = reduced_df.index
                    baseline_fitter = Baseline(x_data=x)
                    f = reduced_df['HRR (W/g)']

                    out = baseline_fitter.imodpoly(f, poly_order = 2, num_std = 1, max_iter = 1000, return_coef = True)
                    g = out[0] # Baseline
                    h = f-g
                    reduced_df[f'{col_name}_alt'] = h
                    reduced_df.dropna(inplace=True)
                    data_df = pd.concat([data_df, reduced_df], axis = 1)

                    data_array_alt = reduced_df[f'{col_name}_alt'].to_numpy()

                    # hoc_df_html.at['Heat of Combustion [ALT] (MJ/kg)', col_name] = (integrate.trapz(y=data_array_alt, x=time_array)) / 1000

        corrected_data = data_df.loc[:,all_col_names]

        plot_data_df.loc[:,'HRR_mean'] = corrected_data.mean(axis = 1)
        plot_data_df.loc[:,'HRR_std'] = corrected_data.std(axis = 1)

        plot_mean_data(plot_data_df)

    else:
        continue

    hoc_df_html = hoc_df_html.round(decimals=2)
    hoc_df_html['Mean'] = hoc_df_html.mean(axis=1).round(decimals=2)
    hoc_df_html['Std. Dev.'] = hoc_df_html.std(axis=1).round(decimals=2)

    hoc_df_html.index.rename('Value',inplace=True)
    hoc_df_html = hoc_df_html.reset_index()
    hoc_df_html.to_html(f'{data_dir}{material}/MCC/{material}_MCC_Heats_of_Combustion.html',index=False,border=0)
    mat_status_df.loc[material, 'MCC_HoC'] = True

    plot_dir = f'{save_dir}{material}/MCC/'

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    format_and_save_plot(f'{plot_dir}{material}_MCC_HRR.html')
    mat_status_df.loc[material, 'MCC_HRR'] = True

mat_status_df.to_csv('Utilities/material_status.csv', index_label = 'material')
print()

