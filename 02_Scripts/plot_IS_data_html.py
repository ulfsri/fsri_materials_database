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


def plot_mean_data(df):
    color_dict = {'MEAS':'black', 'REF': 'red','BL': 'blue'}
    label_dict = {'MEAS':'Mean Sample Measurement', 'REF': 'Reference Measurement','BL': 'Baseline Measurement'}
    for i in ['MEAS', 'REF', 'BL']:

        y_upper =  df[f'{i}_mean'] + 2*df[f'{i}_std']
        y_lower = df[f'{i}_mean'] - 2*df[f'{i}_std']
        try:
            fig.add_trace(go.Scatter(x=np.concatenate([df.index,df.index[::-1]]),y=pd.concat([y_upper,y_lower[::-1]]),
                fill='toself',hoveron='points',fillcolor='lightgray',line=dict(color='lightgray'),name='2'+ "\u03C3"))
        except:
            continue
        fig.add_trace(go.Scatter(x=df.index,y=df[f'{i}_mean'], marker=dict(color=color_dict[i], size=8),name=label_dict[i]))


    return()

def format_and_save_plot(file_loc):
    fig.update_layout(xaxis_title='Wavelength (nm)', font=dict(size=18))
    fig.update_layout(yaxis_title='Reflection Signal (-)', title ='Reflection Signal')

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

for d in os.scandir(data_dir):
    data = pd.DataFrame()
    data_df = pd.DataFrame()
    material = d.path.split('/')[-1]
    print(f'{material} IS')
    ylims = [0,0]
    xlims = [0,0]
    fig = go.Figure()
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

        plot_mean_data(data_df)

        plot_dir = f'../03_Charts/{material}/FTIR/IS/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        format_and_save_plot(f'{plot_dir}{material}_IS_Reflection.html')
