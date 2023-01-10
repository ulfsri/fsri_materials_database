# Heat flow meter html data processing script
#   by: ULRI's Fire Safety Research Institute
#   Questions? Submit them here: https://github.com/ulfsri/fsri_materials_database/issues

# ***************************** Usage Notes *************************** #
# - Script outputs as a function of temperature                         #
#   -  HTML Graphs dir: /03_Charts/{Material}/HFM                       #
#      Graphs: Wet and Dry Thermal Conductivity and Heat Capacity       #
#                                                                       #
#      HTML Tables dir: /01_Data/{Material}/HFM                         #
#      Tables: Wet and Dry Thermal Conductivity and Heat Capacity       #
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

    if file_loc.split('.')[-2].split('_')[-1].lower() == 'conductivity':
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

    # print()

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    material = d
    k_wet = []
    k_dry = []
    if os.path.isdir(f'{data_dir}{d}/HFM/'):
        print(f'{material} Thermal Conductivity')
        for f in sorted(glob.iglob(f'{data_dir}{d}/HFM/*.tst')):
            if 'conductivity' in str(f).lower():
                if 'Dry' in f:
                    k_dry.append(f)
                else:
                    k_wet.append(f)
            else:
                continue
        dirs = [k_wet, k_dry]
        k_plot_data = pd.DataFrame()
        for coll in dirs:
            k_df = pd.DataFrame()

            for file_path in coll:
                if 'conductivity' in str(file_path).lower() and '.csv' not in str(file_path).lower():
                    clean_file(file_path)
                    start_line = search_string_in_file(f'{file_path}_TEMP.tst', 'Results Table -- SI Units')
                    header_line = start_line+1
                    temp_df = pd.read_csv(f'{file_path}_TEMP.tst', sep = '\t', header = header_line, usecols = [1,2,3,4], index_col = 'Mean Temp', engine='python', skip_blank_lines = False, encoding = 'UTF-16')
                    os.remove(f'{file_path}_TEMP.tst')
                    temp_df.dropna(axis=0, inplace = True)

                    temp_df.index = temp_df.index.astype('float64')
                    temp_df_ind = temp_df.index.to_series()
                    temp_df.index = temp_df_ind.round(0)

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

                k_plot_data.at[i,f'{f_str[-4]}_mean'] = round(i_mean, 3)
                k_plot_data.at[i,f'{f_str[-4]}_std'] = round(i_std, 3)

        for cond in ['Wet', 'Dry']:

            k_plot_data_cond = k_plot_data.filter(regex=cond)

            if k_plot_data_cond.empty:
                continue

            fig = go.Figure()

            plot_mean_data(k_plot_data_cond)

            plot_dir = f'../03_Charts/{material}/HFM/'

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            format_and_save_plot(f'{plot_dir}{material}_HFM_{cond}_conductivity.html',material)

            k_plot_data_cond = k_plot_data_cond.rename(columns = {f'{cond}_mean': 'Conductivity Mean (W/m-K)', f'{cond}_std': 'Standard Deviation (W/m-K)'})
            k_plot_data_cond.index.names = ['Temperature (C)']
            k_plot_data_cond = k_plot_data_cond.reset_index()
            k_plot_data_cond.to_html(f'{data_dir}{material}/HFM/{material}_HFM_{cond}_conductivity.html',index=False,border=0)

print()
for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    material = d
    temp_df = pd.DataFrame()
    density_df = pd.DataFrame(index = ['mean', 'std'])
    if os.path.isdir(f'{data_dir}{d}/HFM/'):
        for f in sorted(glob.iglob(f'{data_dir}{d}/HFM/*.csv')):
            if 'Density' in f and 'Summary' not in f and 'Ignition' not in f:
                temp_density_series = pd.read_csv(f, index_col = 0).squeeze()
                f_list = str(f).replace('.csv', '').split('_')
                f_str = f_list[-2] + '_' + f_list[-1]
                temp_df.at['Density', f_str] = temp_density_series['Sample Density [kg/m3]']

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
    c_wet = []
    c_dry = []
    if os.path.isdir(f'{data_dir}{d}/HFM/'):
        file_counter = 0
        for f in sorted(glob.iglob(f'{data_dir}{d}/HFM/*.tst')):
            if 'heatcapacity' in str(f).lower():
                file_counter += 1
                if 'Dry' in f:
                    c_dry.append(f)
                else:
                    c_wet.append(f)

        if file_counter == 0:
            print('empty')
            continue
        dirs = [c_wet, c_dry]
        c_plot_data = pd.DataFrame()
        for coll in dirs:
            c_df = pd.DataFrame()
            for file_path in coll:
                if 'heatcapacity' in str(file_path).lower() and '.csv' not in str(file_path).lower():
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

                c_plot_data.at[i,f'{f_str[-4]}_mean'] = round(i_mean / density_df.at['mean', f_str[-4]], 0)
                c_plot_data.at[i,f'{f_str[-4]}_std'] = round(i_std / density_df.at['mean', f_str[-4]], 0)

    for cond in ['Wet', 'Dry']:

        c_plot_data_cond = c_plot_data.filter(regex=cond)
        if c_plot_data_cond.empty:
            continue

        fig = go.Figure()
        plot_mean_data(c_plot_data_cond)

        plot_dir = f'../03_Charts/{material}/HFM/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        format_and_save_plot(f'{plot_dir}{material}_HFM_{cond}_specific_heat.html',material)

        c_plot_data_cond = c_plot_data_cond.rename(columns = {f'{cond}_mean': 'Specific Heat Mean (J/kg-K)', f'{cond}_std': 'Standard Deviation (J/kg-K)'})
        c_plot_data_cond.index.names = ['Temperature (C)']
        c_plot_data_cond = c_plot_data_cond.reset_index()
        c_plot_data_cond.to_html(f'{data_dir}{material}/HFM/{material}_HFM_{cond}_specific_heat.html', index=False,border=0)
