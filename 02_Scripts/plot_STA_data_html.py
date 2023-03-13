# Simultaneous thermal analyzer html data processing script
#   by: ULRI's Fire Safety Research Institute
#   Questions? Submit them here: https://github.com/ulfsri/fsri_materials_database/issues

# ***************************** Usage Notes *************************** #
# - Script outputs as a function of temperature and heating rate        #
#   -  HTML Graphs dir: /03_Charts/{Material}/N2                        #
#      Graphs: Apparent Heat Capacity, DSC Derivative, Heat Flow Rate,  #
#      Normalized Mass, Normalized Mass Loss Rate                       #
#                                                                       #
#      HTML Tables dir: /01_Data/{Material}/                            #
#      Tables: Melting Temperature Table                                #
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
from scipy import integrate
import plotly.graph_objects as go
import git
from pybaselines import Baseline, utils

def apply_savgol_filter(raw_data, deriv=0):

    window_raw = int((raw_data.count())/40)
    window = int(np.ceil(window_raw) // 2 * 2 + 1)

    if window < 6:
        poly_order = 3
    else:
        poly_order = 5

    raw_data = raw_data.dropna().loc[0:]
    converted_data = savgol_filter(raw_data,window,poly_order, deriv=deriv)
    filtered_data = pd.Series(converted_data, index=raw_data.index.values)
    return(filtered_data.loc[0:])

def plot_mean_data(df):

    hr_dict = {'3K_min':'red', '10K_min':'green', '30K_min':'blue'}

    for i in hr_dict.keys():
        try:
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
        except:
            continue
    return()

def format_and_save_plot(inc, file_loc):
    axis_dict = {'Mass': 'Normalized Mass', 'MLR': 'Normalized MLR (1/s)', 'Flow': 'Heat Flow Rate (W/g)', 'Cp': 'Apparent Heat Capacity (J/g-K)', 'd': 'DSC Derivative (W/g-K)'}
    keyword = file_loc.split('.html')[0].split('_')[-1]

    fig.update_layout(xaxis_title='Temperature (&deg;C)', font=dict(size=18))
    fig.update_layout(yaxis_title=axis_dict[keyword])
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


data_dir = '../01_Data/'
save_dir = '../03_Charts/'

plot_dict = {'Normalized Mass':'Mass', 'Normalized MLR':'MLR', 'Heat Flow Rate':'Heat_Flow', 'Apparent Heat Capacity':'Cp', 'DSC_deriv':'d'}

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    material = d
    melt_temp = []
    melt_onset = []
    melt_enth = []
    if os.path.isdir(f'{data_dir}{d}/STA/'):
        print(material + ' STA')
        plot_data_df = pd.DataFrame()
        for d_ in os.scandir(f'{data_dir}{d}/STA/N2/'):
            data_df = pd.DataFrame()
            reduced_df = pd.DataFrame()
            for f in sorted(glob.iglob(f'{d_.path}/*.csv')):
                HR = d_.path.split('/')[-1]
                if 'Meta' in f:
                    continue
                else:
                    # import data for each test
                    fid = f.split('/')[-1]
                    fid_meta = fid.replace('Data', 'Meta')
                    f_meta = f.replace(fid, fid_meta)

                    data_temp_df = pd.read_csv(f, header = 0)
                    meta_temp_df = pd.read_csv(f_meta).squeeze()
                    meta_col_df = meta_temp_df.filter(regex='EXPORT').squeeze()

                    mass_ind = meta_col_df.str.find('SAMPLE MASS', start = 0).idxmax()
                    m0 = float(meta_temp_df.iloc[mass_ind, 1])

                    data_temp_df['Temp (C)']  = data_temp_df.filter(regex='Temp', axis='columns')

                    data_temp_df['time (s)'] = data_temp_df.filter(regex='Time', axis='columns')
                    data_temp_df['time (s)'] = (data_temp_df['time (s)']-data_temp_df.loc[0,'time (s)'])*60

                    data_temp_df['Heating rate (K/s)'] = np.gradient(data_temp_df['Temp (C)'], data_temp_df['time (s)'])

                    data_temp_df['Mass/mg'] = m0 + data_temp_df.filter(regex='Mass', axis='columns')
                    data_temp_df['nMass'] = data_temp_df['Mass/mg']/data_temp_df.at[0,'Mass/mg']

                    data_temp_df['Normalized MLR (1/s)'] = -np.gradient(data_temp_df['nMass'], data_temp_df['time (s)'])
                    data_temp_df['Normalized MLR (1/s)'] = apply_savgol_filter(data_temp_df['Normalized MLR (1/s)'])

                    test_list = [i for i in data_temp_df.columns.to_list() if 'mW/mg' in i] # determine if DSC is given in mW or mW/mg

                    if not test_list:
                        data_temp_df['DSC/(mW/mg)'] = data_temp_df.filter(regex='DSC', axis='columns')/m0
                    else:
                        data_temp_df['DSC/(mW/mg)'] = data_temp_df.filter(regex='DSC', axis='columns')

                    data_temp_df['Apparent Heat Capacity (J/g-K)'] = data_temp_df['DSC/(mW/mg)']/data_temp_df['Heating rate (K/s)']

                    col_name = f.split('.csv')[0].split('_')[-1]

                    data_temp_df['Temp (C)'] = data_temp_df['Temp (C)'].round(decimals = 1)

                    min_lim = data_temp_df['Temp (C)'].iloc[1] - ((data_temp_df['Temp (C)'].iloc[1])%1)
                    max_lim = data_temp_df['Temp (C)'].iloc[-1] - ((data_temp_df['Temp (C)'].iloc[-1])%1)

                    reduced_df = data_temp_df.loc[:,['Temp (C)', 'time (s)', 'nMass', 'Normalized MLR (1/s)', 'DSC/(mW/mg)', 'Apparent Heat Capacity (J/g-K)']]

                    # Re-index data

                    reduced_df.set_index('Temp (C)', inplace = True)
                    reduced_df = reduced_df.loc[51:]
                    reduced_df = reduced_df[~reduced_df.index.duplicated(keep='first')]

                    reduced_df = reduced_df.reindex(reduced_df.index.union(np.arange(51, (max_lim+0.1), 0.1)))
                    reduced_df = reduced_df.astype('float64')
                    reduced_df.index = reduced_df.index.astype('float64')
                    reduced_df = reduced_df.interpolate(method='cubic')

                    reduced_df = reduced_df.loc[np.arange(51, (max_lim + 0.1), 0.1)]

                    reduced_df['Normalized Mass'] = reduced_df.pop('nMass')
                    reduced_df['Heat Flow Rate (W/g)'] = reduced_df.pop('DSC/(mW/mg)')
                    reduced_df = reduced_df[~reduced_df.index.duplicated(keep='first')]

                    reduced_df['DSC_deriv'] = apply_savgol_filter(reduced_df['Heat Flow Rate (W/g)'], deriv=1)

                    # Determine peak melting temperature, temperature at onset of melting, and heat of melting

                    max_mlr = reduced_df['Normalized MLR (1/s)'].max()
                    mlr_threshold = 0.1*max_mlr
                    max_d_dsc = reduced_df['DSC_deriv'].max()
                    d_dsc_threshold = 0.3*max_d_dsc

                    reduced_df.dropna(axis='index', how='any', inplace = True)

                    signs = np.sign(reduced_df['DSC_deriv']).diff().ne(0)
                    signs_list = signs.index[signs].tolist()

                    signs_list = [round(i,1) for i in signs_list]
                    reduced_df.index = [round(i,1) for i in reduced_df.index]

                    for i in signs_list:
                        i = round(i, 1)
                        j = round(i-3, 1)
                        k = round(i+3, 1)
                        if i < 60 or i > 400:
                            continue
                        elif reduced_df.abs().loc[j,'DSC_deriv'] > d_dsc_threshold and reduced_df.abs().loc[k,'DSC_deriv'] > d_dsc_threshold:
                            if reduced_df.abs().loc[i,'Normalized MLR (1/s)'] < mlr_threshold:
                                melt_temp.append(i)
                                peak_temp = i

                                df_temp = reduced_df.loc[(i-75):(i+75)].copy() # +/- 75 C is arbitrary, but appears to work well for the polynomial baseline fit

                                x = df_temp.index
                                baseline_fitter = Baseline(x_data=x)
                                f = df_temp['Heat Flow Rate (W/g)']

                                out = baseline_fitter.imodpoly(f, poly_order = 3, num_std = 1, max_iter = 1000, return_coef = True)
                                g = out[0] # Baseline
                                h = f-g

                                idx = np.argwhere(np.diff(np.sign(h))).flatten()
                                idx_temp = df_temp.index[idx].to_list()
                                inter_temp = np.array([(i - peak_temp) for i in idx_temp])
                                inter_temp_sign = np.sign(inter_temp)
                                sign_change = ((np.roll(inter_temp_sign, 1) - inter_temp_sign) != 0).astype(int)
                                sign_change[0] = 0
                                idx_sign = np.argmax(sign_change)
                                onset_temp = idx_temp[idx_sign-1] # onset of melting peak / lower limit of integration for heat of melting
                                melt_onset.append(onset_temp)
                                melt_return = idx_temp[idx_sign] # return to baseline from melting peak / upper limit of integration for heat of melting

                                df_temp['DSC_corr'] = h

                                melt_peak_df = df_temp.loc[onset_temp:melt_return]
                                melting_enthalpy = integrate.trapz(melt_peak_df['DSC_corr'], melt_peak_df['time (s)'])
                                melt_enth.append(melting_enthalpy)

                    if data_df.empty:
                        data_df = reduced_df
                    else:
                        data_df = pd.concat([data_df, reduced_df], axis = 1)

            for m in plot_dict.keys():
                data_sub = data_df.filter(regex = m)
                plot_data_df.loc[:,f'{m} {HR} mean'] = data_df.filter(regex = m).mean(axis = 1)
                plot_data_df.loc[:,f'{m} {HR} std'] = data_df.filter(regex = m).std(axis = 1)

        # calculate mean melting temperature
        if len(melt_temp) > 2:
            html_df = pd.DataFrame()
            html_df = pd.DataFrame(index = ['Peak Melting Temperature (C)', 'Temperature at Onset of Melting (C)', 'Enthalpy of Melting (J/g)'], columns = ['Mean', 'Std. Dev.'])

            html_df.at['Peak Melting Temperature (C)', 'Mean'] = np.mean(np.array(melt_temp)).round(1)
            html_df.at['Peak Melting Temperature (C)', 'Std. Dev.'] = np.std(np.array(melt_temp)).round(1)

            html_df.at['Temperature at Onset of Melting (C)', 'Mean'] = np.mean(np.array(melt_onset)).round(1)
            html_df.at['Temperature at Onset of Melting (C)', 'Std. Dev.'] = np.std(np.array(melt_onset)).round(1)

            html_df.at['Enthalpy of Melting (J/g-K)', 'Mean'] = np.mean(np.array(melt_enth)).round(1)
            html_df.at['Enthalpy of Melting (J/g-K)', 'Std. Dev.'] = np.std(np.array(melt_enth)).round(1)

            html_df.index.rename('Value',inplace=True)
            html_df = html_df.reset_index()
            html_df.to_html(f'{data_dir}{material}/STA/{material}_STA_Analysis_Melting_Temp_Table.html',index=False,border=0)

        plot_dir = f'../03_Charts/{material}/STA/N2/'

        plot_inc = {'Mass': 0.2, 'MLR': 0.001, 'Heat_Flow': 0.5, 'Cp': 0.5, 'd': 0.1}

        for m in plot_dict.keys():    
            fig = go.Figure()

            plot_data = plot_data_df.filter(regex = m)

            min_lim = math.ceil(plot_data.index[0])
            max_lim = plot_data.index[-1] - ((plot_data.index[-1])%1)

            plot_data = plot_data.loc[np.arange(min_lim, max_lim, 1)] # Downsample data to every 1 degree C

            plot_mean_data(plot_data)

            inc = plot_inc[plot_dict[m]]

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            suffix = plot_dict[m]
            format_and_save_plot(inc, f'{plot_dir}{material}_STA_{suffix}.html')

    else:
        continue