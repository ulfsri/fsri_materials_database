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

data_dir = '../01_Data/'

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]
    print(f'{material} Thermal Conductivity')
    k_wet = []
    k_dry = []
    if d.is_dir():
        for d_ in os.scandir(d):
            if d_.is_dir() and 'HFM' in d_.path:
                for f in os.scandir(d_):
                    if 'conductivity' in str(f.path).lower() and '.html' not in str(f.path).lower() and '.csv' not in str(f.path).lower():
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
                if 'conductivity' in str(file_path).lower():
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

            k_plot_data_cond = k_plot_data_cond.rename(columns = {f'{cond}_mean': 'Conductivity Mean (W/m-K)', f'{cond}_std': 'Standard Deviation (W/m-K)'})
            k_plot_data_cond.index.names = ['Temperature (C)']
            k_plot_data_cond.to_csv(f'{data_dir}{material}/HFM/{material}_HFM_{cond}_conductivity.csv', index_label = 'Temperature (C)')

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]

    temp_df = pd.DataFrame()
    density_df = pd.DataFrame(index = ['mean', 'std'])
    if d.is_dir():
        for d_ in os.scandir(d):
            if d_.is_dir() and 'HFM' in d_.path:
                for f in os.scandir(d_):
                    if 'Density' in f.path and 'Summary' not in f.path:
                        temp_density_series = pd.read_csv(f, index_col = 0).squeeze()
                        f_list = str(f.path).replace('.csv', '').split('_')
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

    density_df.to_csv(f'{data_dir}{material}/HFM/{material}_HFM_Density_Summary.csv')

    print(f'{material} Heat Capacity')
    c_wet = []
    c_dry = []
    if d.is_dir():
        file_counter = 0
        for d_ in os.scandir(d):
            if d_.is_dir() and 'HFM' in d_.path:
                for f in os.scandir(d_):
                    if 'heatcapacity' in str(f.path).lower() and '.html' not in str(f.path).lower() and '.csv' not in str(f.path).lower():
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
                if 'heatcapacity' in str(file_path).lower():
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

        c_plot_data_cond = c_plot_data_cond.rename(columns = {f'{cond}_mean': 'Specific Heat Mean (J/kg-K)', f'{cond}_std': 'Standard Deviation (J/kg-K)'})
        c_plot_data_cond.index.names = ['Temperature (C)']
        c_plot_data_cond.to_csv(f'{data_dir}{material}/HFM/{material}_HFM_{cond}_specific_heat.csv', index_label = 'Temperature (C)')

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]
    print(material)
    kpc_df = pd.Series(dtype='float64')

    try:
        density_df = pd.read_csv(f'{data_dir}{material}/HFM/{material}_HFM_Density_Summary.csv', index_col = 0)
        conductivity_df = pd.read_csv(f'{data_dir}{material}/HFM/{material}_HFM_Wet_conductivity.csv', index_col = 'Temperature (C)')
        capacity_df = pd.read_csv(f'{data_dir}{material}/HFM/{material}_HFM_Wet_specific_heat.csv', index_col = 'Temperature (C)')
    except:
        # print('NO WET DATA')
        try:
            density_df = pd.read_csv(f'{data_dir}{material}/HFM/{material}_HFM_Density_Summary.csv', index_col = 0)
            conductivity_df = pd.read_csv(f'{data_dir}{material}/HFM/{material}_HFM_Dry_conductivity.csv', index_col = 'Temperature (C)')
            capacity_df = pd.read_csv(f'{data_dir}{material}/HFM/{material}_HFM_Dry_specific_heat.csv', index_col = 'Temperature (C)')
        except:
            # print('NO DRY DATA')
            continue

    try:
        kpc_df.at['Thermal Conductivity (W/m-K)'] = conductivity_df['Conductivity Mean (W/m-K)'].iloc[1]
        kpc_df.at['Heat Capacity (J/kg-K)'] = capacity_df['Specific Heat Mean (J/kg-K)'].iloc[3]
        try:
            kpc_df.at['Density (kg/m3)'] = density_df.loc['mean', 'Wet']
        except:
            kpc_df.at['Density (kg/m3)'] = density_df.loc['mean', 'Dry']
    except:
        continue

    try:    
        kpc_df.to_csv(f'{data_dir}{material}/Cone/{material}_Ignition_Temp_Properties.csv', header=False)
    except:
        kpc_df.to_csv(f'{data_dir}{material}/HFM/{material}_Ignition_Temp_Properties.csv', header=False)
