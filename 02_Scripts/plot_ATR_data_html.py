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
import subprocess

label_size = 20
tick_size = 18
line_width = 2
legend_font = 10
fig_width = 10
fig_height = 6

def plot_mean_data(df):

    y_upper = df['mean'] + 2*df['std']
    y_lower = df['mean'] - 2*df['std']

    fig.add_trace(go.Scatter(x=np.concatenate([df.index,df.index[::-1]]),y=pd.concat([y_upper,y_lower[::-1]]),
        fill='toself',hoveron='points',fillcolor='lightgray',line=dict(color='lightgray'),name='2'+ "\u03C3"))
    fig.add_trace(go.Scatter(x=df.index, y=df['mean'], marker=dict(color='black', size=8),name='Mean'))


    return()

def format_and_save_plot(file_loc):

    fig.update_layout(xaxis_title='Wavelength (nm)', font=dict(size=18))
    fig.update_layout(yaxis_title='ATR Signal (-)', title ='ATR Signal')

    #Get github hash to display on graph
    label = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    fig.add_annotation(dict(font=dict(color='black',size=15),
                                        x=1,
                                        y=1.02,
                                        showarrow=False,
                                        text="Repository Version: " + label,
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
    print(f'{material} ATR')
    ylims = [0,0]
    xlims = [0,0]
    fig = go.Figure()
    if d.is_dir():
        if os.path.isdir(f'{d.path}/FTIR'):
            for f in os.scandir(f'{d.path}/FTIR/ATR/'):
                temp_df = pd.read_csv(f, header = None) # index_col = 0
                temp_df.rename(columns = {0:'wavenumber', 1: 'signal'}, inplace=True)
                temp_df['wavelength'] = 10000000/temp_df.iloc[:,0] # wavelength in nm
                data = pd.concat([data, temp_df], axis = 1)
        else:
            continue

        data_df.loc[:,'mean'] = data.groupby(by=data.columns, axis=1).mean().loc[:,'signal']
        data_df.loc[:,'std'] = data.groupby(by=data.columns, axis=1).std().loc[:,'signal']
        data_df.loc[:,'wavelength'] = data.groupby(by=data.columns, axis=1).mean().loc[:,'wavelength']

        data_df.set_index('wavelength', inplace=True)
        plot_mean_data(data_df)

        plot_dir = f'../03_Charts/{material}/FTIR/ATR/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        format_and_save_plot(f'{plot_dir}{material}_ATR.html')
