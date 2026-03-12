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
from scipy.signal import find_peaks
from scipy import integrate
from scipy.interpolate import make_smoothing_spline
import plotly.graph_objects as go
import git
from pybaselines import Baseline, utils

def apply_savgol_filter(raw_data, deriv=0):

    raw_data = raw_data.dropna()

    if len(raw_data) < 11:
        return raw_data.copy()   # do NOT filter small signals

    window = max(11, int(len(raw_data)/40)//2*2+1)
    poly_order = min(5, window-2)

    filtered = savgol_filter(raw_data, window, poly_order, deriv=deriv)
    return pd.Series(filtered, index=raw_data.index)

def smooth(y, window=51, poly=3):
    y = np.asarray(y).ravel()
    if window >= len(y):
        return y
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window, poly)

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

def safe_mean_std(values):
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], np.nan

    return np.mean(values), np.std(values)

def as_numpy_1d(x):
    """Robust conversion to contiguous float numpy array."""
    import numpy as np

    if hasattr(x, "to_numpy"):
        x = x.to_numpy()

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        x = np.ravel(x)

    # remove non-finite values
    mask = np.isfinite(x)
    return x[mask]

def resample_data(T, Y, dT=1.0):

    # force pure numpy positional arrays
    T = np.asarray(T, dtype=float).reshape(-1)
    Y = np.asarray(Y, dtype=float).reshape(-1)

    # remove nonfinite
    mask = np.isfinite(T) & np.isfinite(Y)
    T = T[mask]
    Y = Y[mask]

    if len(T) < 5:
        raise ValueError("Too few points after cleaning")

    # sort by temperature
    order = np.argsort(T)
    T = T[order]
    Y = Y[order]

    # remove duplicate temperatures
    T_unique, unique_idx = np.unique(T, return_index=True)
    Y_unique = Y[unique_idx]

    # build uniform grid
    Tmin = np.ceil(T_unique.min())
    Tmax = np.floor(T_unique.max())

    if Tmax <= Tmin:
        raise ValueError("Temperature range collapsed")

    Tgrid = np.arange(Tmin, Tmax, dT)

    # interpolate
    Ygrid = np.interp(Tgrid, T_unique, Y_unique)

    return Tgrid, Ygrid

def detect_threshold_onset(
        T, m,
        mlr_fraction=0.03,
        mass_fraction=0.01,
        min_temp=120,
        sustain_points=8):

    m_s = smooth(m)
    mlr = -np.gradient(m_s, T)

    # --- Robust peak ---
    mlr_max = np.percentile(mlr, 99)
    mlr_threshold = mlr_fraction * mlr_max
    mlr_max_idx = np.argmax(mlr)

    # --- Mass thresholds ---
    total_mass_loss = m_s[0] - m_s[-1]
    cumulative_loss = m_s[0] - m_s
    mass_threshold = mass_fraction * total_mass_loss

    onset_idx = None
    end_idx = None

    # -----------------------
    # Find onset (rising edge)
    # -----------------------
    for i in range(len(T) - sustain_points):

        if T[i] < min_temp:
            continue

        cond1 = mlr[i] >= mlr_threshold
        cond2 = cumulative_loss[i] >= mass_threshold

        # sustained above threshold
        cond3 = np.all(mlr[i:i+sustain_points] >= mlr_threshold)

        if cond1 and cond2 and cond3:
            onset_idx = i
            break

    if onset_idx is None:
        return None, None, mlr

    # -----------------------
    # Find falling edge
    # -----------------------
    for j in range(onset_idx + sustain_points, len(T) - sustain_points):

        if j < mlr_max_idx:
            continue

        # detect sustained drop below threshold
        below = np.all(mlr[j:j+sustain_points] < mlr_threshold)

        # confirm downward slope (true falling edge)
        slope = mlr[j] - mlr[j-1]

        if below and slope < 0:
            end_idx = j
            break

    if end_idx is None:
        # fallback: last index above threshold
        above_indices = np.where(mlr >= mlr_threshold)[0]
        if len(above_indices) > 0:
            end_idx = above_indices[-1]

    T_on = T[onset_idx] if onset_idx is not None else None
    T_end = T[end_idx] if end_idx is not None else None
    mlr_on = mlr[onset_idx] if onset_idx is not None else None

    return T_on, T_end, mlr_on

def format_value(v, row_name, mean_exponent=None):

    if row_name in {
        'Mean Normalized MLR at Onset (1/s)',
        'Std. Dev. Normalized MLR at Onset (1/s)'
    }:
        
        # If this is the mean row, determine exponent
        if mean_exponent is None:
            if v == 0:
                return "0.00e+00"
            mean_exponent = int(np.floor(np.log10(abs(v))))
        
        scaled = v / (10 ** mean_exponent)
        return f"{scaled:.2f}e{mean_exponent:+03d}"

    # Other formatting rules
    elif row_name in {
        'Mean Onset Temperature (C)',
        'Std. Dev. Onset Temperature (C)',
        'Mean Temperature at end of Decomposition (C)',
        'Std. Dev. Temperature at end of Decomposition (C)'
    }:
        return f"{v:.0f}"

    elif row_name in {
        'Mean Mass Fraction at Onset',
        'Std. Dev. Mass Fraction at Onset'
    }:
        return f"{v:.3f}"

    else:
        return f"{v:.1f}"

def build_display_table(df):

    display_df = df.copy().astype(object)

    mean_mlr = 'Mean Normalized MLR at Onset (1/s)'
    std_mlr  = 'Std. Dev. Normalized MLR at Onset (1/s)'

    for col in df.columns:

        if mean_mlr in df.index and std_mlr in df.index:

            mean_val = df.loc[mean_mlr, col]
            std_val  = df.loc[std_mlr, col]

            if pd.notna(mean_val) and mean_val != 0:

                exponent = int(np.floor(np.log10(abs(mean_val))))

                # format mean
                scaled_mean = mean_val / (10 ** exponent)
                display_df.loc[mean_mlr, col] = (
                    f"{scaled_mean:.2f}e{exponent:+03d}"
                )

                # format std (only if not NaN)
                if pd.notna(std_val):
                    scaled_std = std_val / (10 ** exponent)
                    display_df.loc[std_mlr, col] = (
                        f"{scaled_std:.2f}e{exponent:+03d}"
                    )
                else:
                    display_df.loc[std_mlr, col] = "NaN"

            else:
                display_df.loc[mean_mlr, col] = "NaN"
                display_df.loc[std_mlr, col]  = "NaN"

    for row in display_df.index:

        # Onset + End Temperatures → no decimals
        if row in {
            'Mean Onset Temperature (&deg;C)',
            'Std. Dev. Onset Temperature (&deg;C)',
            'Mean Temperature at end of Decomposition (&deg;C)',
            'Std. Dev. Temperature at end of Decomposition (&deg;C)'
        }:
            display_df.loc[row] = df.loc[row].map(
                lambda v: f"{v:.0f}" if pd.notna(v) else "NaN"
            )

        # Mass Fraction → 2 decimals
        elif row in {
            'Mean Mass Fraction at Onset',
            'Std. Dev. Mass Fraction at Onset'
        }:
            display_df.loc[row] = df.loc[row].map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "NaN"
            )

    return display_df

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

plot_dict = {'Normalized Mass':'Mass', 'Normalized MLR':'MLR', 'Heat Flow Rate':'Heat_Flow', 'Apparent Heat Capacity':'Cp', 'DSC_deriv':'d'}

mlr_threshold = 0.05
mass_threshold = 0.02

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    material = d
    melt_temp = []
    melt_onset = []
    melt_enth = []

    if os.path.isdir(f'{data_dir}{d}/STA/'):
        print(material + ' STA')
        plot_data_df = pd.DataFrame()
        onset_df= pd.DataFrame()
        for d_ in os.scandir(f'{data_dir}{d}/STA/N2/'):
            data_df = pd.DataFrame()
            reduced_df = pd.DataFrame()

            onset_temp_list = []
            return_temp_list = []
            onset_mlr_list = []
            onset_nmass_list = []
            HR = d_.path.split('/')[-1]
            for f in sorted(glob.iglob(f'{d_.path}/*.csv')):
                if 'Meta' in f:
                    continue
                else:
                    # import data for each test
                    fid = f.split('/')[-1]
                    fid_meta = fid.replace('Data', 'Meta')
                    f_meta = f.replace(fid, fid_meta)

                    data_temp_df = pd.read_csv(f, header = 0)
                    meta_temp_df = pd.read_csv(f_meta).squeeze()

                    meta_col_df = meta_temp_df.iloc[:,0].squeeze()

                    mass_ind = meta_col_df.str.find('SAMPLE MASS', start = 0).idxmax()
                    m0 = float(meta_temp_df.iloc[mass_ind, 1])

                    data_temp_df['Temp (C)']  = data_temp_df.filter(regex='Temp', axis='columns').squeeze()

                    data_temp_df['time (s)'] = (data_temp_df.filter(regex='Time', axis='columns').squeeze() - data_temp_df.filter(regex='Time', axis='columns').squeeze().iloc[0]) * 60
                    data_temp_df = data_temp_df[~data_temp_df['time (s)'].duplicated(keep='first')].reset_index(drop=True)
                    dt = data_temp_df['time (s)'].diff().abs()
                    data_temp_df = data_temp_df[(dt > 1e-6) | (dt.isna())].reset_index(drop=True)

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

                    reduced_df = data_temp_df.loc[:,['Temp (C)', 'time (s)', 'nMass', 'Normalized MLR (1/s)', 'DSC/(mW/mg)', 'Apparent Heat Capacity (J/g-K)']].copy()

                    # Re-index data

                    reduced_df.set_index('Temp (C)', inplace = True)
                    # reduced_df = reduced_df.loc[51:]
                    # reduced_df = reduced_df[~reduced_df.index.duplicated(keep='first')]

                    # reduced_df = reduced_df.reindex(reduced_df.index.union(np.arange(51, (max_lim+0.1), 0.1)))
                    # reduced_df = reduced_df.astype('float64')
                    # reduced_df.index = reduced_df.index.astype('float64')
                    # reduced_df = reduced_df.interpolate(method='slinear')

                    # reduced_df = reduced_df.loc[np.arange(51, (max_lim + 0.1), 0.1)]

                    tmin = max(51, reduced_df.index.min())
                    tmax = reduced_df.index.max()

                    reduced_df.dropna(inplace = True)
                    new_index, _ = resample_data(reduced_df.index.to_numpy(), reduced_df['time (s)'].to_numpy(), dT=0.1)

                    interp_df = pd.DataFrame(index=new_index)
                    for col in ['time (s)', 'nMass', 'Normalized MLR (1/s)', 'DSC/(mW/mg)', 'Apparent Heat Capacity (J/g-K)']:
                        T, interp_df[col] = resample_data(reduced_df.index, reduced_df[col], dT = 0.1)
                    reduced_df = interp_df

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

                    reduced_df_abs = reduced_df.abs()

                    for i in signs_list:
                        i = round(i, 1)
                        j = round(i-3, 1)
                        k = round(i+3, 1)
                        if i < 85 or i > 400:
                            continue
                        elif reduced_df_abs.loc[j,'DSC_deriv'] > d_dsc_threshold and reduced_df_abs.loc[k,'DSC_deriv'] > d_dsc_threshold:
                            if reduced_df_abs.loc[i,'Normalized MLR (1/s)'] < mlr_threshold:
                                melt_temp.append(i)
                                peak_temp = i

                                df_temp = reduced_df.loc[(i-75):(i+75)].copy() # +/- 75 C is arbitrary, but appears to work well for the polynomial baseline fit

                                # x = np.ascontiguousarray(df_temp.index.to_numpy(dtype=np.float64))
                                # y = np.ascontiguousarray(df_temp['Heat Flow Rate (W/g)'].to_numpy(dtype=np.float64))

                                x = as_numpy_1d(df_temp.index)
                                y = as_numpy_1d(df_temp['Heat Flow Rate (W/g)'])

                                baseline_fitter = Baseline(x_data=x)

                                baseline, params = baseline_fitter.imodpoly(
                                    y,
                                    poly_order=3,
                                    num_std=1,
                                    max_iter=250,
                                    return_coef=True
                                )

                                corrected = np.ascontiguousarray(y - baseline)

                                g = pd.Series(baseline, index=df_temp.index)
                                h = pd.Series(corrected, index=df_temp.index)

                                idx = np.argwhere(np.diff(np.sign(h.to_numpy()))).flatten()
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

                                melt_peak_df = df_temp.loc[onset_temp:melt_return].copy()
                                melting_enthalpy = integrate.trapezoid(melt_peak_df['DSC_corr'], melt_peak_df['time (s)'])
                                melt_enth.append(melting_enthalpy)

                    # Calculate temperature at onset of decomposition #
                    while True:
                        try:

                            mass_df = reduced_df['Normalized Mass'] = apply_savgol_filter(reduced_df['Normalized Mass'])

                            m = mass_df.values
                            T = mass_df.index.values

                            onset_temp, return_temp, onset_mlr = detect_threshold_onset(T, m, mlr_fraction=0.05, mass_fraction=0.02, min_temp=120)

                            onset_temp_list.append(onset_temp)
                            return_temp_list.append(return_temp)
                            onset_mlr_list.append(onset_mlr)
                            onset_nmass_list.append(reduced_df.at[onset_temp, 'Normalized Mass'])

                            break

                        except Exception as e: 
                            print(f"Warning: {e}")
                            break

                    if data_df.empty:
                        data_df = reduced_df
                        concat_list = [data_df]
                    else:
                        concat_list.append(reduced_df)

            data_df = pd.concat(concat_list, axis = 1)

            for m in plot_dict.keys():
                subset_df = data_df.filter(regex = m)
                if subset_df.shape[1] == 0:
                    continue
                plot_data_df.loc[:,f'{m} {HR} mean'] = subset_df.mean(axis = 1)
                if subset_df.shape[1] > 1:
                    plot_data_df.loc[:,f'{m} {HR} std'] = subset_df.std(axis = 1)
                else:
                    plot_data_df.loc[:,f'{m} {HR} std'] = 0

            # populate onset_df dataframe filled with data pertinent to onset of decomposition

            HR_display_dict = {'10K_min': '10 &deg;C/min', '30K_min': '30 &deg;C/min', '3K_min': '3 &deg;C/min'}

            m, s = safe_mean_std(onset_temp_list)
            onset_df.at['Mean Onset Temperature (&deg;C)', HR_display_dict[HR]] = m
            onset_df.at['Std. Dev. Onset Temperature (&deg;C)', HR_display_dict[HR]] = s

            m, s = safe_mean_std(onset_mlr_list)
            onset_df.at['Mean Normalized MLR at Onset (1/s)', HR_display_dict[HR]] = m
            onset_df.at['Std. Dev. Normalized MLR at Onset (1/s)', HR_display_dict[HR]] = s

            m, s = safe_mean_std(onset_nmass_list)
            onset_df.at['Mean Mass Fraction at Onset', HR_display_dict[HR]] = m
            onset_df.at['Std. Dev. Mass Fraction at Onset', HR_display_dict[HR]] = s

            m, s = safe_mean_std(return_temp_list)
            onset_df.at['Mean Temperature at end of Decomposition (&deg;C)', HR_display_dict[HR]] = m
            onset_df.at['Std. Dev. Temperature at end of Decomposition (&deg;C)', HR_display_dict[HR]] = s

        display_df = build_display_table(onset_df)

        if '3 &deg;C/min' in display_df.columns and '10 &deg;C/min' in display_df.columns and '30 &deg;C/min' in display_df.columns:
            display_df = display_df.loc[:,['3 &deg;C/min', '10 &deg;C/min', '30 &deg;C/min']]
        elif '10 &deg;C/min' in display_df.columns and '30 &deg;C/min' in display_df.columns:
            display_df = display_df.loc[:,['10 &deg;C/min', '30 &deg;C/min']]
        else:
            pass

        display_df.to_html(f'{data_dir}{material}/STA/{material}_STA_Decomposition_Onset_Temp_Table.html',escape=False,index=True,border=0)

        # calculate mean melting temperature
        if len(melt_temp) > 2:
            html_df = pd.DataFrame()
            html_df = pd.DataFrame(index = ['Peak Melting Temperature (C)', 'Temperature at Onset of Melting (C)', 'Enthalpy of Melting (J/g-K)'], columns = ['Mean', 'Std. Dev.'])

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
            plot_mean_data(plot_data)

            inc = plot_inc[plot_dict[m]]

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            suffix = plot_dict[m]
            format_and_save_plot(inc, f'{plot_dir}{material}_STA_{suffix}.html')

    else:
        continue