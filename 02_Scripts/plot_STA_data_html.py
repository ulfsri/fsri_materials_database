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
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import git

label_size = 20
tick_size = 18
line_width = 2
legend_font = 10
fig_width = 10
fig_height = 6

def apply_savgol_filter(raw_data):

    window_raw = int((raw_data.count())/40)
    window = int(np.ceil(window_raw) // 2 * 2 + 1)

    if window < 6:
        poly_order = 3
    else:
        poly_order = 5

    raw_data = raw_data.dropna().loc[0:]
    converted_data = savgol_filter(raw_data,window,poly_order)
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

def plot_mean_data(df):

    hr_dict = {'3K_min':'red', '10K_min':'green', '30K_min':'blue'}

    for i in hr_dict.keys():
        mean_df = df.filter(regex = 'mean')
        mean_hr_df_temp = mean_df.filter(regex = i)
        mean_hr_df = mean_hr_df_temp.dropna(axis = 'index')
        std_df = df.filter(regex = 'std')
        std_hr_df_temp = std_df.filter(regex = i)
        std_hr_df = std_hr_df_temp.dropna(axis = 'index')

        y_upper = mean_hr_df.iloc[:,0] + 2*std_hr_df.iloc[:,0]
        y_lower = mean_hr_df.iloc[:,0] - 2*std_hr_df.iloc[:,0]

        i_str = i.replace('_','/')

        fig.add_trace(go.Scatter(x=mean_hr_df.index, y=mean_hr_df.iloc[:,0], marker=dict(color=hr_dict[i], size=8),name=i_str))
        fig.add_trace(go.Scatter(x=y_lower.index,y=y_lower,fill=None, mode='lines', line_color= hr_dict[i], hoveron='points',name='-2'+ "\u03C3"))
        fig.add_trace(go.Scatter(x=y_upper.index,y=y_upper,
            fill='tonexty',hoveron='points',line_color=hr_dict[i],mode='lines',opacity=0.25,name='+2'+ "\u03C3"))
    return()

def format_and_save_plot(inc, file_loc):
    axis_dict = {'Mass': 'Normalized Mass', 'MLR': 'Normalized MLR (1/s)', 'Flow': 'Heat Flow Rate (W/g)'}
    keyword = file_loc.split('.html')[0].split('_')[-1]

    fig.update_layout(xaxis_title='Temperature (&deg;C)', font=dict(size=18))
    fig.update_layout(yaxis_title=axis_dict[keyword], title ='Simultaneous Thermal Analysis')

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
    print()

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

plot_dict = {'Normalized Mass':'Mass', 'Normalized MLR':'MLR', 'Heat Flow Rate':'Heat_Flow'}

for d in os.scandir(data_dir):
    material = d.path.split('/')[-1]
    if material == '.DS_Store':
        continue
    plot_data_df = pd.DataFrame()
    print(f'{material} STA')
    if d.is_dir():
        if os.path.isdir(f'{d.path}/STA/'):
            for d_ in os.scandir(f'{d.path}/STA/N2/'):
                data_df = pd.DataFrame()
                reduced_df = pd.DataFrame()
                for f in glob.iglob(f'{d_.path}/*.csv'):
                    HR = d_.path.split('/')[-1]
                    if 'Meta' in f or '.DS_Store' in f:
                        continue
                    else:
                        # import data for each test
                        print(f)

                        data_temp_df = pd.read_csv(f, header = 0)
                        data_temp_df['Temp (C)']  = data_temp_df.filter(regex='Temp', axis='columns')
                        data_temp_df['time (s)'] = data_temp_df.filter(regex='Time', axis='columns')
                        data_temp_df['Mass/%'] = data_temp_df.filter(regex='Mass', axis='columns')
                        data_temp_df['DSC/(mW/mg)'] = data_temp_df.filter(regex='DSC', axis='columns')
                        
                        data_temp_df['Mass/%'] = data_temp_df['Mass/%']/data_temp_df.loc[0,'Mass/%']
                        data_temp_df['time (s)'] = (data_temp_df['time (s)']-data_temp_df.loc[0,'time (s)'])*60
                        data_temp_df['Normalized MLR (1/s)'] = -data_temp_df['Mass/%'].diff()/data_temp_df['time (s)'].diff()
                        data_temp_df['Normalized MLR (1/s)'] = apply_savgol_filter(data_temp_df['Normalized MLR (1/s)'])

                        # data_temp_df = pd.read_csv(f, header = 0)
                        # data_temp_df.rename(columns = {'##Temp./°C':'Temp (C)', 'Time/min':'time (s)'}, inplace = True)
                        # data_temp_df['Mass/%'] = data_temp_df['Mass/%']/data_temp_df.loc[0,'Mass/%']
                        # data_temp_df['time (s)'] = (data_temp_df['time (s)']-data_temp_df.loc[0,'time (s)'])*60
                        # data_temp_df['Normalized MLR (1/s)'] = -data_temp_df['Mass/%'].diff()/data_temp_df['time (s)'].diff()
                        # data_temp_df['Normalized MLR (1/s)'] = apply_savgol_filter(data_temp_df['Normalized MLR (1/s)'])

                        col_name = f.split('.csv')[0].split('_')[-1]

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

    # plot_data_df.to_csv(f'{data_dir}{material}/STA/N2/TEST_html.csv')

    plot_dir = f'../03_Charts/{material}/STA/N2/'

    plot_inc = {'Mass': 0.2, 'MLR': 0.001, 'Heat_Flow': 0.5}

    for m in plot_dict.keys():    
        fig = go.Figure()

        plot_data = plot_data_df.filter(regex = m)
        plot_mean_data(plot_data)

        inc = plot_inc[plot_dict[m]]

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        suffix = plot_dict[m]
        format_and_save_plot(inc, f'{plot_dir}{material}_STA_{suffix}.html')