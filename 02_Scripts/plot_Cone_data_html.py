# Cone calorimeter html data processing script
#   by: ULRI's Fire Safety Research Institute
#   Questions? Submit them here: https://github.com/ulfsri/fsri_materials_database/issues

# ***************************** Usage Notes *************************** #
# - Script outputs as a function of heat flux                           #
#   -  HTML Graphs dir: /03_Charts/{Material}/Cone                      #
#      Graphs: Extinction_Coefficient, Heat Release Rate Per Unit Area, #
#      Mass Loss Rate, Specific Extinction Area, Smoke Production Rate  #
#                                                                       #
#      HTML Tables dir: /01_Data/{Material}/Cone                        #
#      Tables: Heat Release Per Unit Area, CO Table, Soot Table         #
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
from scipy.integrate import trapezoid
import plotly.graph_objects as go
import git

plot_all = True
if not plot_all: 
    print('plot_all is set to False, so any materials with existing html output files will be skipped')

### Fuel Properties ###
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

label_size = 20
tick_size = 18
line_width = 2
legend_font = 10
fig_width = 10
fig_height = 6

def create_1plot_fig():
    # Define figure for the plot
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
    #plt.subplots_adjust(left=0.08, bottom=0.3, right=0.92, top=0.95)

    # Reset values for x & y limits
    x_min, x_max, y_min, y_max = 0, 0, 0, 0

    return(fig, ax1)

def format_and_save_plot(quantity, file_loc,m):

    label_dict = {'HRRPUA': 'HRRPUA (kW/m<sup>2</sup>)', 'MLR': 'Mass Loss Rate (g/s)', 'EHC':'Effective Heat of Combustion (MJ/kg)' , 'SPR': 'Smoke Production Rate (1/s)', 'SEA': 'Specific Extinction Area', 'Extinction Coefficient': 'Extinction Coefficient (1/m)', 'CO': 'CO Yield (g/g)', 'Soot': 'Soot Yield (g/g)'}

    if quantity == 'MLR':
        fig.update_yaxes(rangemode='nonnegative')

    fig.update_layout(xaxis_title='Time (s)', font=dict(size=18))
    fig.update_layout(yaxis_title=label_dict[quantity])
    # fig.update_layout(autosize=True, width=513, height=450,margin=dict(l=25,r=25,b=40,t=40))
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

    fig.write_html(file_loc.replace(' ','_'),include_plotlyjs="cdn")
    plt.close()

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

# initialize material status dataframe
if os.path.isfile('Utilities/material_status.csv'):
    mat_status_df = pd.read_csv('Utilities/material_status.csv', index_col = 'material')
else:
    mat_status_df = pd.DataFrame(columns = ['Wet_cp', 'Dry_cp', 'Wet_k', 'Dry_k', 'STA_MLR', 'CONE_MLR_25', 'CONE_MLR_50', 'CONE_MLR_75', 'CONE_HRRPUA_25', 'CONE_HRRPUA_50', 'CONE_HRRPUA_75', 'CO_Yield', 'MCC_HRR', 'Soot_Yield', 'MCC_HoC', 'Cone_HoC', 'HoR', 'HoG', 'MCC_Ign_Temp', 'Melting_Temp', 'Emissivity', 'Full_JSON', "Picture"])

    for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
        if os.path.isdir(f'{data_dir}/{d}'):
            material = d

            r = np.empty((23, ))
            r[:] = np.nan
            mat_status_df.loc[material, :] = r
    mat_status_df.fillna(False, inplace=True)


hf_list_default = ['25', '50', '75']
quant_list = ['HRRPUA', 'MLR', 'SPR', 'Extinction Coefficient'] #'EHC', 'CO Yield', 'Soot Yield','SEA',

y_max_dict = {'HRRPUA':500, 'MLR':1, 'SPR':5, 'Extinction Coefficient':2} #'EHC':50000, 'CO':0.1, 'Soot':0.1,'SEA':1000,
y_inc_dict = {'HRRPUA':100, 'MLR':0.2, 'SPR':1, 'Extinction Coefficient':0.5} #'EHC':10000, 'CO':0.02, 'Soot':0.02,'SEA':200,

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".") and f != 'README.md'), key=str.lower):
    df_dict = {}
    material = d
    summary_df = pd.DataFrame()
    output_df = pd.DataFrame()
    co_df = pd.DataFrame()
    soot_df = pd.DataFrame()
    # if material != 'EPDM_Membrane': continue
    if not plot_all:
        output_exists = False
        for c in ['CONE_MLR_25', 'CONE_MLR_50', 'CONE_MLR_75', 'CONE_HRRPUA_25', 'CONE_HRRPUA_50', 'CONE_HRRPUA_75', 'CO_Yield', 'Cone_HoC', 'Soot_Yield']: 
            if mat_status_df.loc[material, c]: output_exists = True
        if output_exists: 
            # print(f'Skipping {material} Cone --- plot_all is False and output charts exist')
            continue

    if os.path.isdir(f'{data_dir}{d}/Cone/'):
        print(material + ' Cone')
        data_df = pd.DataFrame()
        reduced_df = pd.DataFrame()
        if os.path.isfile(f'{data_dir}{d}/Cone/hf_list.csv'):
            hf_list =  pd.read_csv(f'{data_dir}{d}/Cone/hf_list.csv') # for parsing hf outside of base set of ranges
        else:
            hf_list = hf_list_default
        for f in sorted(glob.iglob(f'{data_dir}{d}/Cone/*.csv')):
            if 'scan' in f.lower():
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
                
                # import notes to get sample surface area
                try:
                    notes_df = pd.read_csv(f'{data_dir}{d}/Cone/{material}_Cone_Notes.csv', index_col = 0)
                    rep = f.split('_')[-1].replace('.csv','')
                    HF = f.split('_')[-3].replace('Scan', '')
                    notes_ind = f'{HF}_{rep}'
                    surf_area_mm2 = notes_df.loc[notes_ind, 'Surface Area (mm^2)']
                except:
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
                # data_temp_df.loc[:,'Smoke Comp Norm'] = (data_temp_df.loc[:, 'Smoke Comp']/data_temp_df.at['Baseline','Smoke Comp'])
                # data_temp_df.loc[:,'Smoke Meas Norm'] = (data_temp_df.loc[:, 'Smoke Meas']/data_temp_df.at['Baseline','Smoke Meas'])
                data_temp_df['THR'] = 0.25*data_temp_df['HRRPUA'].cumsum()/1000
                data_temp_df['MLR_grad'] = -np.gradient(data_temp_df['Sample Mass'], 0.25)
                data_temp_df['MLR'] = apply_savgol_filter(data_temp_df['MLR_grad'])
                data_temp_df['MLR'][data_temp_df['MLR'] > 5] = 0
                # data_temp_df['MLR'] = np.zeros(len(data_temp_df['Sample Mass']))

                data_temp_df['EHC'] = data_temp_df['HRR']/data_temp_df['MLR'] # kW/(g/s) -> MJ/kg
                # data_temp_df['Extinction Coefficient'] = data_temp_df['Ext Coeff'] - data_temp_df.at['Baseline','Ext Coeff']
                data_temp_df['Smoke Validation'] = (data_temp_df.at['Baseline','Smoke Meas'] == data_temp_df.loc[:,'Smoke Meas'])
                if data_temp_df['Smoke Validation'].iloc[1:].any():
                    pass
                else:
                    data_temp_df['Extinction Coefficient'] = (1/0.11)*np.log(data_temp_df.at['Baseline','Smoke Meas']/data_temp_df.loc[:,'Smoke Meas']) # 1/m
                    # data_temp_df['Soot Mass Concentration'] = (laser_wl*data_temp_df['Extinction Coefficient']*smoke_density)/c # kg/m3
                    data_temp_df['Soot Mass Concentration'] = data_temp_df['Extinction Coefficient']/avg_ext_coeff # kg/m3
                    data_temp_df['Soot Mass Fraction'] = data_temp_df['Soot Mass Concentration']/air_density(data_temp_df['Smoke TC']) # kg/kg
                    data_temp_df['Soot Mass Flow'] = data_temp_df['Soot Mass Fraction']*data_temp_df['EDF'] # kg/s
                    data_temp_df['Soot Yield'] = data_temp_df['Soot Mass Flow']/data_temp_df['MLR'] # kg/kg
                    data_temp_df['SPR'] = (data_temp_df.loc[:,'Extinction Coefficient'] * data_temp_df.loc[:,'Volumetric Flow'])/float(scalar_data_series.at['SURF AREA'])
                    data_temp_df.loc[data_temp_df.loc[:,'SPR'] < 0,'SPR'] = 0
                    data_temp_df['SEA'] = (1000*data_temp_df.loc[:,'Volumetric Flow']*data_temp_df.loc[:,'Extinction Coefficient'])/data_temp_df['MLR']

                data_temp_df['CO Mass Fraction'] = data_temp_df['CO Meter']*(CO_density(data_temp_df['Smoke TC'])/air_density(data_temp_df['Smoke TC']))
                data_temp_df['CO Mass Flow'] = data_temp_df['CO Mass Fraction']*data_temp_df['EDF'] # kg/s
                data_temp_df['CO Yield'] = data_temp_df['CO Mass Flow']/data_temp_df['MLR'] # kg/kg

                try:
                    df_dict[label] = data_temp_df[['Time', 'HRRPUA', 'MLR', 'EHC', 'SPR', 'SEA', 'Extinction Coefficient', 'CO Yield', 'Soot Yield']].copy()
                except:
                    df_dict[label] = data_temp_df[['Time', 'HRRPUA', 'MLR', 'EHC', 'CO Yield']].copy()

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
                end_mass = float(data_temp_df.loc[str(end_ind),'Sample Mass'])

                if float(data_temp_df.loc[str(1),'Sample Mass']) - float(data_temp_df.loc[end_ind,'Sample Mass']) > float(scalar_data_series['SPECIMEN MASS']):
                    try:
                        sample_mass_inc = data_temp_df.loc[str(2):end_ind,'Sample Mass'].diff().abs() > 1
                        mass_discont_list = sample_mass_inc.index[sample_mass_inc == True].tolist()
                        mass_discont_list_test = [abs(int(end_ind) - int(i)) for i in mass_discont_list]
                        if mass_discont_list_test:
                            if int(min(mass_discont_list_test)) < 150: # this is an arbitrary threshold that appears to work well - filters out dicontinuities early in tests
                                end_ind = int(min(mass_discont_list))-4 # -4 is an arbitrary number that works well - this ensure that if the holder was removed, we go back 1 second for the final mass 
                                if end_ind < 0:
                                    end_ind = int(min(mass_discont_list))
                        end_mass = float(data_temp_df.loc[str(end_ind),'Sample Mass'])
                        if float(data_temp_df.loc[str(1),'Sample Mass']) - end_mass < 0:
                            end_mass = float(data_temp_df.loc[str(1),'Sample Mass']) - float(scalar_data_series['SPECIMEN MASS'])

                    except:
                        end_mass = float(data_temp_df.loc[str(1),'Sample Mass']) - float(scalar_data_series['SPECIMEN MASS'])

                mass_lost = ign_mass-end_mass
                ml_10 = ign_mass - 0.1*mass_lost
                ml_90 = ign_mass - 0.9*mass_lost
                ml_10_ind = data_temp_df['Sample Mass'].sub(ml_10).abs().idxmin()
                ml_90_ind = data_temp_df['Sample Mass'].sub(ml_90).abs().idxmin()

                x = data_temp_df.loc[ml_10_ind:ml_90_ind, 'Time'].to_numpy()

                try:
                    soot_prod = (trapezoid(data_temp_df.loc[ml_10_ind:ml_90_ind,'Soot Mass Flow'], x = x))*1000 # g
                    soot_df.at['Soot Yield (g/g)', label] = soot_prod/(0.8*mass_lost)
                except:
                    pass

                co_prod = (trapezoid(data_temp_df.loc[ml_10_ind:ml_90_ind,'CO Mass Flow'], x = x))*1000 # g
                co_df.at['CO Yield (g/g)', label] = co_prod/(0.8*mass_lost)

                summary_df.at['Time to Sustained Ignition (s)', label] = scalar_data_series.at['TIME TO IGN']
                summary_df.at['Peak HRRPUA (kW/m2)', label] = float("{:.2f}".format(max(data_temp_df['HRRPUA'])))
                summary_df.at['Time to Peak HRRPUA (s)', label] = data_temp_df.loc[data_temp_df['HRRPUA'].idxmax(), 'Time'] - float(scalar_data_series.at['TIME TO IGN'])

                ign_index = data_temp_df.index[data_temp_df['Time'] == float(scalar_data_series.at['TIME TO IGN'])][0]
                t60 = str(int(ign_index) + 240)
                t180 = str(int(ign_index) + 720)
                t300 = str(int(ign_index) + 1200)

                try: summary_df.at['Average HRRPUA over 60 seconds (kW/m2)', label] = float("{:.2f}".format(np.mean(data_temp_df.loc[ign_index:t60,'HRRPUA'])))
                except: summary_df.at['Average HRRPUA over 60 seconds (kW/m2)', label] = math.nan

                try: summary_df.at['Average HRRPUA over 180 seconds (kW/m2)', label] = float("{:.2f}".format(np.mean(data_temp_df.loc[ign_index:t180,'HRRPUA'])))
                except: summary_df.at['Average HRRPUA over 180 seconds (kW/m2)', label] = math.nan

                try: summary_df.at['Average HRRPUA over 300 seconds (kW/m2)', label] = float("{:.2f}".format(np.mean(data_temp_df.loc[ign_index:t300,'HRRPUA'])))
                except: summary_df.at['Average HRRPUA over 300 seconds (kW/m2)', label] = math.nan

                summary_df.at['Total Heat Released (MJ/m2)', label] = float("{:.2f}".format(data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'THR']))
                total_mass_lost = data_temp_df.at['1','Sample Mass'] - data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'Sample Mass']
                holder_mass = data_temp_df.at['1','Sample Mass'] - float(scalar_data_series.at['SPECIMEN MASS'])
                summary_df.at['Avg. Effective Heat of Combustion (MJ/kg)', label] = float("{:.2f}".format(((data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'THR'])*surf_area_m2)/(total_mass_lost/1000)))
                summary_df.at['Initial Mass (g)', label] = scalar_data_series.at['SPECIMEN MASS']
                summary_df.at['Final Mass (g)', label] = float("{:.2f}".format(data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'Sample Mass'] - holder_mass))
                summary_df.at['Mass at Ignition (g)', label] = float("{:.2f}".format(data_temp_df.at[ign_index,'Sample Mass'] - holder_mass))

                t10 = data_temp_df['Sample Mass'].sub(data_temp_df.at['1','Sample Mass'] - 0.1*total_mass_lost).abs().idxmin()
                t90 = data_temp_df['Sample Mass'].sub(data_temp_df.at['1','Sample Mass'] - 0.9*total_mass_lost).abs().idxmin()

                summary_df.at['Avg. Mass Loss Rate [10% to 90%] (g/m2s)', label] = float("{:.2f}".format(np.mean(data_temp_df.loc[t10:t90,'MLR']/surf_area_m2)))

        for n in quant_list:
            for m in hf_list:
                fig = go.Figure()
                for key, value in df_dict.items():
                    rep_str = key.split('_')[-1]
                    if m in key:
                        plot_df = df_dict[key].filter(regex = n)
                        try:
                            plot_data(plot_df, rep_str)
                        except:
                            continue
                plot_dir = f'../03_Charts/{material}/Cone/'

                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                
                format_and_save_plot(n, f'{plot_dir}{material}_Cone_{n}_{m}.html',m)
                if f'CONE_{n}_{m}' in list(mat_status_df.columns): mat_status_df.loc[material, f'CONE_{n}_{m}'] = True

    else:
        continue

    summary_df = summary_df.round(1)
    summary_df.sort_index(axis=1, inplace=True)
    summary_df.to_csv(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_Data.csv', float_format='%.1f')

    co_html_df = pd.DataFrame(index = hf_list, columns = ['Mean CO Yield [g/g]', 'CO Yield Std. Dev. [g/g]'])
    soot_html_df = pd.DataFrame(index = hf_list, columns = ['Mean Soot Yield [g/g]', 'Soot Yield Std. Dev. [g/g]'])
    hoc_html_df = pd.DataFrame(index = hf_list, columns = ['Mean Effective Heat of Combustion [MJ/kg]', 'Effective Heat of Combustion Std. Dev. [MJ/kg]'])

    for hf in hf_list:
        html_df = output_df.filter(like=hf)
        html_df = html_df.rename(columns=lambda x: hf +' kW/m\u00b2 ' + x.split('_')[-1])
        html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_HRRPUA_Table_{hf}.html', float_format='%.2f', encoding='UTF-8', border=0)

        co_hf_df = co_df.filter(like=hf)
        co_html_df.loc[hf,'Mean CO Yield [g/g]'] = np.around(co_hf_df.mean(axis=1).to_numpy()[0], decimals=3)
        co_html_df.loc[hf, 'CO Yield Std. Dev. [g/g]'] = np.around(co_hf_df.std(axis=1).to_numpy()[0], decimals=3)
        co_html_df.index.rename('Incident Heat Flux [kW/m\u00b2]',inplace=True)

        soot_hf_df = soot_df.filter(like=hf)
        soot_hf_df = soot_hf_df.drop(columns=[col for col in soot_hf_df.columns if soot_hf_df[col].lt(0).any()])
        soot_html_df.loc[hf,'Mean Soot Yield [g/g]'] = np.around(soot_hf_df.mean(axis=1).to_numpy()[0], decimals=3)
        soot_html_df.loc[hf, 'Soot Yield Std. Dev. [g/g]'] = np.around(soot_hf_df.std(axis=1).to_numpy()[0], decimals=3)
        soot_html_df.index.names = ['Incident Heat Flux [kW/m\u00b2]']

        hoc_df = summary_df.filter(like=hf)
        hoc_df = hoc_df.filter(regex = 'Heat of Combustion', axis = 'index')
        hoc_html_df.loc[hf, 'Mean Effective Heat of Combustion [MJ/kg]'] = np.around(hoc_df.mean(axis=1).to_numpy()[0], decimals=1)
        hoc_html_df.loc[hf, 'Effective Heat of Combustion Std. Dev. [MJ/kg]'] = np.around(hoc_df.std(axis=1).to_numpy()[0], decimals=1)
        hoc_html_df.index.names = ['Incident Heat Flux [kW/m\u00b2]']

    co_html_df = co_html_df.reset_index()
    co_html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_CO_Table.html',index=False, encoding='UTF-8', border=0)
    mat_status_df.loc[material, 'CO_Yield'] = True

    if soot_html_df.isnull().values.any():
        pass
    else:
        soot_html_df = soot_html_df.reset_index()
        soot_html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_Soot_Table.html',index=False, encoding='UTF-8', border=0)
        mat_status_df.loc[material, 'Soot_Yield'] = True

    hoc_html_df = hoc_html_df.reset_index()
    hoc_html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_EHC_Table.html',index=False, encoding='UTF-8', border=0)
    mat_status_df.loc[material, 'Cone_HoC'] = True

mat_status_df.to_csv('Utilities/material_status.csv', index_label = 'material')
print()