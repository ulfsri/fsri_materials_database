# MCC Data Import and Pre-processing
#   by: Mark McKinnon
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
from scipy.integrate import trapezoid

label_size = 20
tick_size = 18
line_width = 2
legend_font = 10
fig_width = 10
fig_height = 6

### Fuel Properties ###
# a =
# b =
# c =
# d =
e = 13100 # [kJ/kg O2] del_hc/r_0
laser_wl = 632.8/10e9 # m
smoke_density = 1100 # kg/m3
c = 7 # average coefficient of smoke extinction
avg_ext_coeff = 8700 # m2/kg  from Mullholland

def apply_savgol_filter(raw_data):

    # raw_data.drop('Baseline', axis = 'index', inplace = True)
    raw_data = raw_data.dropna()
    converted_data = savgol_filter(raw_data,31,3)
    filtered_data = pd.Series(converted_data, index=raw_data.index.values)
    return(filtered_data.iloc[0:])

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

def plot_data(df, rep):

    rep_dict = {'R1': 'black', 'R2': 'blue', 'R3': 'red', 'R4': 'green', 'R5': 'magenta', 'R6': 'cyan'}

    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:,0], marker=dict(color=rep_dict[rep], size=8),name=rep))

    return()

def air_density(temperature):
    # returns density in kg/m3 given a temperature in C
    Pr = 101325
    R_spec = 287.1
    T = temperature+273.15
    # rho = 1.2883 - 4.327e-3*temperature + 8.78e-6*temperature**2
    rho = Pr/(R_spec*T)
    return rho

def CO_density(temperature):
    # returns density in kg/m3 given a temperature in C
    Pr = 101325
    R_spec = 296.8
    T = temperature+273.15
    # rho = 1.2883 - 4.327e-3*temperature + 8.78e-6*temperature**2
    rho = Pr/(R_spec*T)
    return rho

def format_and_save_plot(quantity, file_loc,m):

    label_dict = {'HRRPUA': 'Heat Release Rate per Unit Area (kW/m<sup>2</sup>)', 'MLR': 'Mass Loss Rate (g/s)', 'EHC':'Effective Heat of Combustion (MJ/kg)' , 'SPR': 'Smoke Production Rate (1/s)', 'SEA': 'Specific Extinction Area', 'Extinction Coefficient': 'Extinction Coefficient (1/m)', 'CO': 'CO Yield (g/g)', 'Soot': 'Soot Yield (g/g)'}

    fig.update_layout(xaxis_title='Time (s)', font=dict(size=18))
    fig.update_layout(yaxis_title=label_dict[quantity], title ='Cone Calorimeter at ' + m + ' kW/m<sup>2</sup>')

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

hf_list = ['25', '50', '75']
quant_list = ['HRRPUA', 'MLR', 'SPR', 'SEA', 'Extinction Coefficient'] #'EHC', 'CO Yield', 'Soot Yield'

y_max_dict = {'HRRPUA':500, 'MLR':1, 'SPR':5, 'SEA':1000, 'Extinction Coefficient':2, 'EHC':50000, 'CO':0.1, 'Soot':0.1}
y_inc_dict = {'HRRPUA':100, 'MLR':0.2, 'SPR':1, 'SEA':200, 'Extinction Coefficient':0.5, 'EHC':10000, 'CO':0.02, 'Soot':0.02}

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    df_dict = {}
    material = d
    output_df = pd.DataFrame()
    co_df = pd.DataFrame()
    soot_df = pd.DataFrame()
    notes_df = pd.DataFrame()
    # if d.is_dir():
    if os.path.isdir(f'{data_dir}{d}/Cone/'):
        print(material + ' Cone')
        data_df = pd.DataFrame()
        reduced_df = pd.DataFrame()
        for f in sorted(glob.iglob(f'{data_dir}{d}/Cone/*.csv')):
        # for f in os.scandir(f'{d.path}/Cone/'):
            if 'scalar' in f.lower() or 'cone_analysis_data' in f.lower() or 'cone_notes' in f.lower()or 'hrrpua_table' in f.lower():
                continue
            else:
                label_list = f.split('.csv')[0].split('_')
                label = label_list[-3].split('Scan')[0] + '_' + label_list[-1]
                data_temp_df = pd.read_csv(f, header = 0, skiprows = [1, 2, 3, 4], index_col = 'Names')

                scalar_data_fid = f.replace('Scan','Scalar')
                scalar_data_series = pd.read_csv(scalar_data_fid, index_col = 0).squeeze()

                # Test Notes #
                try:
                    pretest_notes = scalar_data_series.at['PRE TEST CMT']
                except:
                    pretest_notes = ' '
                surf_area_mm2 = 10000
                dims = 'not specified'
                frame = False
                for notes in pretest_notes.split(';'):
                    if 'Dimensions' in notes:
                        dims = []
                        for i in notes.split(' '):
                            try:
                                dims.append(float(i))
                            except: continue
                        surf_area_mm2 = dims[0] * dims[1]
                    elif 'frame' in notes:
                        frame = True
                if frame or '-Frame' in f:
                        surf_area_mm2 = 8836

                surf_area_m2 = surf_area_mm2 / 1000000.0

                # notes_df.at[label, 'Dimensions (mm)'] = str(dims)
                notes_df.at[label, 'Surface Area (mm^2)'] = surf_area_mm2

                notes_df.at[label, 'Pretest'] = pretest_notes
                try:
                    notes_df.at[label, 'Posttest'] = scalar_data_series.at['POST TEST CMT']
                except:
                    notes_df.at[label, 'Posttest'] = ' '

                c_factor = float(scalar_data_series.at['C FACTOR'])

                data_temp_df['O2 Meter'] = data_temp_df['O2 Meter']/100
                data_temp_df['CO2 Meter'] = data_temp_df['CO2 Meter']/100
                data_temp_df['CO Meter'] = data_temp_df['CO Meter']/100

                data_temp_df.loc[:,'EDF'] = ((data_temp_df.loc[:,'Exh Press']/(data_temp_df.loc[:,'Stack TC']+273.15)).apply(np.sqrt)).multiply(c_factor) # Exhaust Duct Flow (m_e_dot) kg/s
                data_temp_df.loc[:,'Volumetric Flow'] = data_temp_df.loc[:,'EDF']/air_density(data_temp_df.loc[:,'Smoke TC']) # Exhaust Duct Flow (V_e_dot)
                # O2_offset = 0.2095 - data_temp_df.at['Baseline', 'O2 Meter']
                # data_temp_df.loc[:,'ODF'] = (0.2095 - data_temp_df.loc[:,'O2 Meter'] + O2_offset) / (1.105 - (1.5*(data_temp_df.loc[:,'O2 Meter'] + O2_offset))) # Oxygen depletion factor with only O2
                data_temp_df.loc[:,'ODF'] = (data_temp_df.at['Baseline', 'O2 Meter'] - data_temp_df.loc[:,'O2 Meter']) / (1.105 - (1.5*(data_temp_df.loc[:,'O2 Meter']))) # Oxygen depletion factor with only O2
                data_temp_df.loc[:,'ODF_ext'] = (data_temp_df.at['Baseline', 'O2 Meter']*(1-data_temp_df.loc[:, 'CO2 Meter'] - data_temp_df.loc[:, 'CO Meter']) - data_temp_df.loc[:, 'O2 Meter']*(1-data_temp_df.at['Baseline', 'CO2 Meter']))/(data_temp_df.at['Baseline', 'O2 Meter']*(1-data_temp_df.loc[:, 'CO2 Meter']-data_temp_df.loc[:, 'CO Meter']-data_temp_df.loc[:, 'O2 Meter'])) # Oxygen Depletion Factor with O2, CO, and CO2
                data_temp_df.loc[:,'HRR'] = 1.10*(e)*data_temp_df.loc[:,'EDF']*data_temp_df.loc[:,'ODF']
                data_temp_df.loc[:,'HRR_ext'] = 1.10*(e)*data_temp_df.loc[:,'EDF']*data_temp_df.at['Baseline', 'O2 Meter']*((data_temp_df.loc[:,'ODF_ext']-0.172*(1-data_temp_df.loc[:,'ODF'])*(data_temp_df.loc[:, 'CO2 Meter']/data_temp_df.loc[:, 'O2 Meter']))/((1-data_temp_df.loc[:,'ODF'])+1.105*data_temp_df.loc[:,'ODF']))
                data_temp_df.loc[:,'HRRPUA'] = data_temp_df.loc[:,'HRR']/float(scalar_data_series.at['SURF AREA'])
                data_temp_df.loc[:,'Smoke Comp Norm'] = (data_temp_df.loc[:, 'Smoke Comp']/data_temp_df.at['Baseline','Smoke Comp'])
                data_temp_df.loc[:,'Smoke Meas Norm'] = (data_temp_df.loc[:, 'Smoke Meas']/data_temp_df.at['Baseline','Smoke Meas'])
                data_temp_df['THR'] = 0.25*data_temp_df['HRRPUA'].cumsum()/1000
                data_temp_df['MLR_grad'] = -np.gradient(data_temp_df['Sample Mass'], 0.25)
                data_temp_df['MLR'] = apply_savgol_filter(data_temp_df['MLR_grad'])
                data_temp_df['MLR'][data_temp_df['MLR'] > 5] = 0
                # data_temp_df['MLR'] = np.zeros(len(data_temp_df['Sample Mass']))

                # # MLR Calculation
                # data_temp_df['MLR'].iloc[0] = ( 25*(data_temp_df['Sample Mass'].iloc[0]) - 48*(data_temp_df['Sample Mass'].iloc[1]) + 36*(data_temp_df['Sample Mass'].iloc[2]) - 16*(data_temp_df['Sample Mass'].iloc[3]) + 3*(data_temp_df['Sample Mass'].iloc[4])) / (12*0.25)
                # data_temp_df['MLR'].iloc[1] = ( 3*(data_temp_df['Sample Mass'].iloc[0]) + 10*(data_temp_df['Sample Mass'].iloc[1]) - 18*(data_temp_df['Sample Mass'].iloc[2]) + 6*(data_temp_df['Sample Mass'].iloc[3]) - (data_temp_df['Sample Mass'].iloc[4])) / (12*0.25)
                # for i in range(2, len(data_temp_df['Sample Mass'])):
                #     if i == len(data_temp_df['Sample Mass'])-2:
                #         data_temp_df['MLR'].iloc[i] = ( -3*(data_temp_df['Sample Mass'].iloc[i]) - 10*(data_temp_df['Sample Mass'].iloc[i-1]) + 18*(data_temp_df['Sample Mass'].iloc[i-2]) - 6*(data_temp_df['Sample Mass'].iloc[i-3]) + (data_temp_df['Sample Mass'].iloc[i-4])) / (12*0.25)
                #     elif i == len(data_temp_df['Sample Mass'])-1:
                #         data_temp_df['MLR'].iloc[i] = ( -25*(data_temp_df['Sample Mass'].iloc[i]) + 48*(data_temp_df['Sample Mass'].iloc[i-1]) - 36*(data_temp_df['Sample Mass'].iloc[i-2]) + 16*(data_temp_df['Sample Mass'].iloc[i-3]) - 3*(data_temp_df['Sample Mass'].iloc[i-4])) / (12*0.25)
                #     else:
                #         data_temp_df['MLR'].iloc[i] = ( -(data_temp_df['Sample Mass'].iloc[i-2]) + 8*(data_temp_df['Sample Mass'].iloc[i-1]) - 8*(data_temp_df['Sample Mass'].iloc[i+1]) + (data_temp_df['Sample Mass'].iloc[i+2])) / (12*0.25)

                data_temp_df['EHC'] = data_temp_df['HRR']/data_temp_df['MLR'] # kW/(g/s) -> MJ/kg
                # data_temp_df['Extinction Coefficient'] = data_temp_df['Ext Coeff'] - data_temp_df.at['Baseline','Ext Coeff']
                data_temp_df['Extinction Coefficient'] = (1/0.11)*np.log(data_temp_df.at['Baseline','Smoke Meas']/data_temp_df.loc[:,'Smoke Meas']) # 1/m
                # data_temp_df['Soot Mass Concentration'] = (laser_wl*data_temp_df['Extinction Coefficient']*smoke_density)/c # kg/m3
                data_temp_df['Soot Mass Concentration'] = data_temp_df['Extinction Coefficient']/avg_ext_coeff # kg/m3
                data_temp_df['Soot Mass Fraction'] = data_temp_df['Soot Mass Concentration']/air_density(data_temp_df['Smoke TC']) # kg/kg
                data_temp_df['Soot Mass Flow'] = data_temp_df['Soot Mass Fraction']*data_temp_df['EDF'] # kg/s
                data_temp_df['Soot Yield'] = data_temp_df['Soot Mass Flow']/data_temp_df['MLR'] # kg/kg
                data_temp_df['CO Mass Fraction'] = data_temp_df['CO Meter']*(CO_density(data_temp_df['Smoke TC'])/air_density(data_temp_df['Smoke TC']))
                data_temp_df['CO Mass Flow'] = data_temp_df['CO Mass Fraction']*data_temp_df['EDF'] # kg/s
                data_temp_df['CO Yield'] = data_temp_df['CO Mass Flow']/data_temp_df['MLR'] # kg/kg

                data_temp_df['SPR'] = (data_temp_df.loc[:,'Extinction Coefficient'] * data_temp_df.loc[:,'Volumetric Flow'])/float(scalar_data_series.at['SURF AREA'])
                data_temp_df['SPR'][data_temp_df['SPR'] < 0] = 0
                data_temp_df['SEA'] = (1000*data_temp_df.loc[:,'Volumetric Flow']*data_temp_df.loc[:,'Extinction Coefficient'])/data_temp_df['MLR']

                df_dict[label] = data_temp_df[['Time', 'HRRPUA', 'MLR', 'EHC', 'SPR', 'SEA', 'Extinction Coefficient', 'CO Yield', 'Soot Yield']].copy()
                df_dict[label].set_index(df_dict[label].loc[:,'Time'], inplace = True)
                df_dict[label] = df_dict[label][df_dict[label].index.notnull()]
                df_dict[label].drop('Time', axis = 1, inplace = True)
                end_time = float(scalar_data_series.at['END OF TEST TIME'])
                num_intervals = (max(df_dict[label].index)-end_time)/0.25
                drop_list = list(np.linspace(end_time, max(df_dict[label].index), int(num_intervals+1)))
                df_dict[label].drop(labels = drop_list, axis = 0, inplace = True)

                # Determine intervals for yield integrals

                ign_time = float(scalar_data_series.at['TIME TO IGN'])
                end_time = float(scalar_data_series.at['END OF TEST TIME'])
                ign_ind = str(int(4 * ign_time + 1))
                end_ind = str(int(4 * end_time + 1))
                ign_mass = float(data_temp_df.loc[ign_ind,'Sample Mass'])
                end_mass = float(data_temp_df.loc[end_ind,'Sample Mass'])
                mass_lost = ign_mass-end_mass
                ml_10 = ign_mass - 0.1*mass_lost
                ml_90 = ign_mass - 0.9*mass_lost
                ml_10_ind = data_temp_df['Sample Mass'].sub(ml_10).abs().idxmin()
                ml_90_ind = data_temp_df['Sample Mass'].sub(ml_90).abs().idxmin()

                x = data_temp_df.loc[ml_10_ind:ml_90_ind, 'Time'].to_numpy()

                soot_prod = (trapezoid(data_temp_df.loc[ml_10_ind:ml_90_ind,'Soot Mass Flow'], x = x))*1000 # g
                soot_df.at['Soot Yield (g/g)', label] = soot_prod/(0.8*mass_lost)
                co_prod = (trapezoid(data_temp_df.loc[ml_10_ind:ml_90_ind,'CO Mass Flow'], x = x))*1000 # g
                co_df.at['CO Yield (g/g)', label] = co_prod/(0.8*mass_lost)

                output_df.at['Peak HRRPUA (kW/m2)', label] = float("{:.2f}".format(max(data_temp_df['HRRPUA'])))
                output_df.at['Time to Peak HRRPUA (s)', label] = data_temp_df.loc[data_temp_df['HRRPUA'].idxmax(), 'Time'] - float(scalar_data_series.at['TIME TO IGN'])

        for n in quant_list:
            for m in hf_list:
                fig = go.Figure()
                for key, value in df_dict.items():
                    rep_str = key.split('_')[-1]
                    if m in key:
                        plot_df = df_dict[key].filter(regex = n)
                        plot_data(plot_df, rep_str)

                inc = y_inc_dict[n]

                # print(n)
                # print(f'{m} kW/m2')
                # print(inc)

                plot_dir = f'../03_Charts/{material}/Cone/'

                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)

                format_and_save_plot(n, f'{plot_dir}{material}_Cone_{n}_{m}.html',m)

    else:
        continue

    co_html_df = pd.DataFrame(index = ['25', '50', '75'], columns = ['Mean CO Yield [g/g]', 'CO Yield Std. Dev. [g/g]'])
    soot_html_df = pd.DataFrame(index = ['25', '50', '75'], columns = ['Mean Soot Yield [g/g]', 'Soot Yield Std. Dev. [g/g]'])

    hf_ranges = ['25','50','75']
    for hf in hf_ranges:
        html_df = output_df.filter(like=hf)
        html_df.columns = html_df.columns.str.replace('HF'+hf+'_', hf+' kW/m"&#xB2;" ')
        html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_HRRPUA_Table_{hf}.html', float_format='%.2f',classes='col-xs-12 col-sm-6')

        co_hf_df = co_df.filter(like=hf)
        # co_html_df.loc[hf, 'Incident Heat Flux [kW/m"&#xB2;"]'] = hf

        co_html_df.loc[hf,'Mean CO Yield [g/g]'] = np.around(co_hf_df.mean(axis=1).to_numpy()[0], decimals=3)
        co_html_df.loc[hf, 'CO Yield Std. Dev. [g/g]'] = np.around(co_hf_df.std(axis=1).to_numpy()[0], decimals=3)
        co_html_df.index.names = ['Incident Heat Flux [kW/m"&#xB2;"]']

        soot_hf_df = soot_df.filter(like=hf)
        soot_hf_df.drop(columns=[col for col in soot_hf_df.columns if soot_hf_df[col].lt(0).any()], inplace=True)

        soot_html_df.loc[hf,'Mean Soot Yield [g/g]'] = np.around(soot_hf_df.mean(axis=1).to_numpy()[0], decimals=3)
        soot_html_df.loc[hf, 'Soot Yield Std. Dev. [g/g]'] = np.around(soot_hf_df.std(axis=1).to_numpy()[0], decimals=3)
        soot_html_df.index.names = ['Incident Heat Flux [kW/m"&#xB2;"]']


    co_html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_CO_Table.html',# float_format='%.3f',
    classes='col-xs-12 col-sm-6')

    if soot_html_df.isnull().values.any():
        pass
    else:
        soot_html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_Soot_Table.html', #float_format='%.3f',
        classes='col-xs-12 col-sm-6')
