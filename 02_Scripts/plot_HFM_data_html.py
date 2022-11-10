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
import git

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
    columns_list = list(df.columns)
    colors = {'Wet':'blue', 'Dry':'red'}
    for i in range(int(len(df.columns)/2)):
        col_ref = i-1
        mean_col = 2*(i-1)
        std_col = (2*i)-1
        data_lab = columns_list[col_ref].split('_')[0]
        c = colors[data_lab]
        fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:,mean_col],error_y=dict(type='data', array=2*df.iloc[:,std_col],),mode='markers',name=data_lab, marker=dict(color=c, size=8)))

    return()

def format_and_save_plot(file_loc,material):

    if file_loc.split('.')[-2].split('_')[-1] == 'Conductivity':
        fig.update_layout(yaxis_title='Thermal Conductivity (W/mK)', title ='Thermal Conductivity')
    else:
        fig.update_layout(yaxis_title='Specific Heat (J/kgK)', title ='Specific Heat')

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

    # Save plot to file
    fig.update_layout(xaxis_title='Temperature (&deg;C)', font=dict(size=18))
    fig.write_html(file_loc,include_plotlyjs="cdn")
    plt.close()

    print()

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]
    print(f'{material} Thermal Conductivity')
    ylims = [0,0]
    xlims = [0,0]
    fig = go.Figure()
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

        plot_mean_data(k_plot_data)

        plot_dir = f'../03_Charts/{material}/HFM/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        format_and_save_plot(f'{plot_dir}{material}_Thermal_Conductivity.html',material)

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]
   
    temp_df = pd.DataFrame() 
    density_df = pd.DataFrame(index = ['mean', 'std'])
    if d.is_dir():
        for d_ in os.scandir(d):
            if d_.is_dir() and 'Density' in d_.path:
                for f in os.scandir(d_):
                    temp_density_series = pd.read_csv(f, index_col = 0).squeeze()
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
    fig = go.Figure()
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

        plot_mean_data(c_plot_data)

        plot_dir = f'../03_Charts/{material}/HFM/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        print('Plotting Chart')
        format_and_save_plot(f'{plot_dir}{material}_Specific_Heat.html',material)
