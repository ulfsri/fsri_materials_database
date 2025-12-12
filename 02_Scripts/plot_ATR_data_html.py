# ATR html data processing script
#   by: ULRI's Fire Safety Research Institute
#   Questions? Submit them here: https://github.com/ulfsri/fsri_materials_database/issues

# ***************************** Usage Notes *************************** #
# - Script outputs as a function of wavelength                          #
#   -  HTML Graphs dir: /03_Charts/{Material}/FTIR/ATR                  #
#      Graphs: ATR Signal                                               #
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

def plot_mean_data(df):

    y_upper = df['mean'] + 2*df['std']
    y_lower = df['mean'] - 2*df['std']

    fig.add_trace(go.Scatter(x=np.concatenate([df.index,df.index[::-1]]),y=pd.concat([y_upper,y_lower[::-1]]),
        fill='toself',hoveron='points',fillcolor='lightgray',line=dict(color='lightgray'),name='2'+ "\u03C3"))
    fig.add_trace(go.Scatter(x=df.index, y=df['mean'], marker=dict(color='black', size=8),name='Mean'))


    return()

def format_and_save_plot(file_loc):

    fig.update_layout(xaxis_title='Wavelength (nm)', font=dict(size=18))
    fig.update_layout(yaxis_title='ATR Signal (-)')
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
    print()

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    if os.path.isdir(f'{data_dir}{d}/FTIR/ATR'):
        if not any(filename.endswith('.tst') for filename in os.listdir(f'{data_dir}{d}/FTIR/ATR')): continue
        data = pd.DataFrame()
        data_df = pd.DataFrame()
        material = d
        print(f'{material} ATR')
        fig = go.Figure()
        for f in sorted(glob.iglob(f'{data_dir}{material}/FTIR/ATR/*.tst')):
            temp_df = pd.read_csv(f, header = None) # index_col = 0
            temp_df.rename(columns = {0:'wavenumber', 1: 'signal'}, inplace=True)
            temp_df['wavelength'] = 10000000/temp_df.iloc[:,0] # wavelength in nm
            data = pd.concat([data, temp_df], axis = 1)
    else: 
        continue

    data_df.loc[:,'mean'] = data.groupby(by=data.columns, axis=1).mean().loc[:,'signal']
    data_df.loc[:,'std'] = data.groupby(by=data.columns, axis=1).std(numeric_only=True).loc[:,'signal']
    data_df.loc[:,'wavelength'] = data.groupby(by=data.columns, axis=1).mean().loc[:,'wavelength']

    data_df.set_index('wavelength', inplace=True)
    plot_mean_data(data_df)

    plot_dir = f'../03_Charts/{material}/FTIR/ATR/'

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    format_and_save_plot(f'{plot_dir}{material}_ATR.html')
