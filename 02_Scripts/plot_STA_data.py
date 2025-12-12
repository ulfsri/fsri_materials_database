# Simultaneous thermal analyzer pdf data processing script
#   by: ULRI's Fire Safety Research Institute
#   Questions? Submit them here: https://github.com/ulfsri/fsri_materials_database/issues

# ***************************** Usage Notes *************************** #
# - Script outputs as a function of temperature and heating rate        #
#   -  PDF Graphs dir: /03_Charts/{Material}/N2                        #
#      Graphs: Apparent Heat Capacity, DSC Derivative, Heat Flow Rate,  #
#      Normalized Mass, Normalized Mass Loss Rate                       #
#                                                                       #
#      CSV Tables dir: /01_Data/{Material}/                            #
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
import git
from pybaselines import Baseline, utils

label_size = 20
tick_size = 18
line_width = 2
legend_font = 10
fig_width = 10
fig_height = 6

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

def create_1plot_fig():
    # Define figure for the plot
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    #plt.subplots_adjust(left=0.08, bottom=0.3, right=0.92, top=0.95)

    # Reset values for x & y limits
    x_min, x_max, y_min, y_max = 0, 0, 0, 0

    return(fig, ax1, x_min, x_max, y_min, y_max)

def plot_mean_data(df):

    hr_dict = {'3K_min':'r', '10K_min':'g', '30K_min':'b'}

    for i in hr_dict.keys():
        try:
            mean_df = df.filter(regex = 'mean')
            mean_hr_df_temp = mean_df.filter(regex = i)
            mean_hr_df = mean_hr_df_temp.dropna(axis = 'index')
            std_df = df.filter(regex = 'std')
            std_hr_df_temp = std_df.filter(regex = i)
            std_hr_df = std_hr_df_temp.dropna(axis = 'index')

            upper_lim = mean_hr_df.iloc[:,0] + 2*std_hr_df.iloc[:,0]
            lower_lim = mean_hr_df.iloc[:,0] - 2*std_hr_df.iloc[:,0]

            i_str = i.replace('_','/')

            ax1.plot(mean_hr_df,color=hr_dict[i],ls='-', marker=None,label = i_str)
            ax1.fill_between(upper_lim.index, lower_lim, upper_lim, color = hr_dict[i], alpha = 0.2)
        except:
            continue

        y_max = upper_lim.max()
        y_min = lower_lim.min()

        x_max = max(df.index)
        x_min = min(df.index)

    return(y_min, y_max, x_min, x_max)

def format_and_save_plot(xlims, ylims, inc, file_loc):
    axis_dict = {'Mass': 'Normalized Mass', 'MLR': 'Normalized MLR (1/s)', 'Flow': 'Heat Flow Rate (W/g)', 'Cp': 'Apparent Heat Capacity (J/g-K)', 'd': 'DSC Derivative (W/g-K)'}
    keyword = file_loc.split('.pdf')[0].split('_')[-1]

    # Set tick parameters
    ax1.tick_params(labelsize=tick_size, length=8, width=0.75, direction = 'inout')

    # Scale axes limits & labels
    ax1.set_ylim(bottom=ylims[0], top=ylims[1])
    ax1.set_xlim(left=xlims[0], right=xlims[1])
    ax1.set_xlabel('Temperature (C)', fontsize=label_size)

    ax1.set_position([0.15, 0.3, 0.77, 0.65])

    y_range_array = np.arange(ylims[0], ylims[1] + inc, inc)
    ax1.set_ylabel(axis_dict[keyword], fontsize=label_size)

    yticks_list = list(y_range_array)

    x_range_array = np.arange(xlims[0], xlims[1] + 50, 50)
    xticks_list = list(x_range_array)

    ax1.set_yticks(yticks_list)
    ax1.set_xticks(xticks_list)

    ax2 = ax1.secondary_yaxis('right')
    ax2.tick_params(axis='y', direction='in', length = 4)
    ax2.set_yticks(yticks_list)
    empty_labels = ['']*len(yticks_list)
    ax2.set_yticklabels(empty_labels)

    ax3 = ax1.secondary_xaxis('top')
    ax3.tick_params(axis='x', direction='in', length = 4)
    ax3.set_xticks(xticks_list)
    empty_labels = ['']*len(xticks_list)
    ax3.set_xticklabels(empty_labels)

    #Get github hash to display on graph
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.commit.hexsha
    short_sha = repo.git.rev_parse(sha, short=True)

    ax1.text(1, 1,'Repository Version: ' + short_sha,
         horizontalalignment='right',
         verticalalignment='bottom',
         transform = ax1.transAxes)

    # Add legend
    handles1, labels1 = ax1.get_legend_handles_labels()

    plt.legend(handles1, labels1, loc = 'upper center', bbox_to_anchor = (0.5, -0.23), fontsize=16,
                handlelength=2, frameon=True, framealpha=1.0, ncol=3)

    # Clean up whitespace padding
    #fig.tight_layout()

    # Save plot to file
    plt.savefig(file_loc)
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
        print(f'{material} STA')
        plot_data_df = pd.DataFrame()
        for d_ in os.scandir(f'{data_dir}{d}/STA/N2/'):
            data_df = pd.DataFrame()
            reduced_df = pd.DataFrame()
            for f in glob.iglob(f'{d_.path}/*.csv'):
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
                                melting_enthalpy = integrate.trapezoid(melt_peak_df['DSC_corr'], melt_peak_df['time (s)'])
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
            melt_df = pd.DataFrame()
            melt_df = pd.DataFrame(index = ['Peak Melting Temperature (C)', 'Temperature at Onset of Melting (C)', 'Enthalpy of Melting (J/g-K)'], columns = ['Mean', 'Std. Dev.'])

            melt_df.at['Peak Melting Temperature (C)', 'Mean'] = np.mean(np.array(melt_temp)).round(1)
            melt_df.at['Peak Melting Temperature (C)', 'Std. Dev.'] = np.std(np.array(melt_temp)).round(1)

            melt_df.at['Temperature at Onset of Melting (C)', 'Mean'] = np.mean(np.array(melt_onset)).round(1)
            melt_df.at['Temperature at Onset of Melting (C)', 'Std. Dev.'] = np.std(np.array(melt_onset)).round(1)

            melt_df.at['Enthalpy of Melting (J/g-K)', 'Mean'] = np.mean(np.array(melt_enth)).round(1)
            melt_df.at['Enthalpy of Melting (J/g-K)', 'Std. Dev.'] = np.std(np.array(melt_enth)).round(1)

            melt_df.to_csv(f'{data_dir}{material}/STA/{material}_STA_Analysis_Melting_Temp_Table.csv')

        plot_dir = f'../03_Charts/{material}/STA/N2/'

        plot_inc = {'Mass': 0.2, 'MLR': 0.001, 'Heat_Flow': 0.5, 'Cp': 0.5, 'd': 0.1}

        for m in plot_dict.keys():
            ylims = [0,0]
            xlims = [0,0]
            fig, ax1, x_min, x_max, y_min, y_max = create_1plot_fig()

            plot_data = plot_data_df.filter(regex = m)
            ymin, ymax, xmin, xmax = plot_mean_data(plot_data)

            y_min = max(ymin, y_min)
            x_min = max(xmin, x_min)
            y_max = max(ymax, y_max)
            x_max = max(xmax, x_max)

            inc = plot_inc[plot_dict[m]]

            ylims[0] = inc * (math.floor(y_min/inc))
            ylims[1] = inc * (math.ceil(y_max/inc))
            xlims[0] = 50 * (math.floor(x_min/50))
            xlims[1] = 50 * (math.ceil(x_max/50))

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            suffix = plot_dict[m]
            format_and_save_plot(xlims, ylims, inc, f'{plot_dir}{material}_STA_{suffix}.pdf')

    else:
        continue
