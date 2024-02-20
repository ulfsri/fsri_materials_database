# Furniture calorimeter html data processing script
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
import plotly.express as px
import git
from PIL import Image

def apply_savgol_filter(raw_data):

    # raw_data.drop('Baseline', axis = 'index', inplace = True)
    raw_data = raw_data.dropna()
    converted_data = savgol_filter(raw_data,51,3)
    filtered_data = pd.Series(converted_data, index=raw_data.index.values)
    return(filtered_data.iloc[0:])

def plot_data(df, rep):

    rep_dict = {'R1': 'black', 'R2': 'blue', 'R3': 'red', 'R4': 'green', 'R5': 'magenta', 'R6': 'cyan'}

    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[0:,0], marker=dict(color=rep_dict[rep], size=8),name=rep))

    return()

def plot_hf_data(df):

    hf_dict = {'HF1': 'black', 'HF2': 'blue', 'HF3': 'red', 'HF4': 'green', 'HF5': 'magenta', 'HF6': 'cyan'}

    for col in df.columns:
        if 'mean' in col:
            hfg_str = col.split('_mean')[0]
            col_str = col.replace('_','/')
            col_std = col.replace('mean', 'std')
            fig.add_trace(go.Scatter(x=df.index, y=df[col], marker=dict(color=hf_dict[hfg_str], size=8),name=hfg_str))
            y_upper = df[col] + 2*df[col_std]
            y_lower = df[col] - 2*df[col_std]
            fig.add_trace(go.Scatter(x=y_lower.index,y=y_lower,fill=None, mode='lines', line_color= hf_dict[hfg_str], hoveron='points',name='-2'+ "\u03C3"))
            fig.add_trace(go.Scatter(x=y_upper.index,y=y_upper,fill='tonexty',hoveron='points',line_color=hf_dict[hfg_str],mode='lines',opacity=0.25,name='+2'+ "\u03C3"))
    hf_legend = Image.open('Utilities/schematics/Furniture_Modeling_HFG.jpg')
    fig.add_layout_image(source = hf_legend, xref = 'x domain', yref = 'y domain', xanchor = 'right', yanchor = 'top', x = 0.95, y = 0.95, sizex = 0.4, sizey = 0.35)

    return()

def plot_plume_temp_data(df):

    TC_dict = {'TCPlume1': 'black', 'TCPlume2': 'blue', 'TCPlume3': 'red'}

    for col in df.columns:
        if 'mean' in col:
            TC_str = col.split('_mean')[0]
            col_str = col.replace('_','/')
            col_std = col.replace('mean', 'std')
            fig.add_trace(go.Scatter(x=df.index, y=df[col], marker=dict(color=TC_dict[TC_str], size=8),name=TC_str))
            y_upper = df[col] + 2*df[col_std]
            y_lower = df[col] - 2*df[col_std]
            fig.add_trace(go.Scatter(x=y_lower.index,y=y_lower,fill=None, mode='lines', line_color= TC_dict[TC_str], hoveron='points',name='-2'+ "\u03C3"))
            fig.add_trace(go.Scatter(x=y_upper.index,y=y_upper,fill='tonexty',hoveron='points',line_color=TC_dict[TC_str],mode='lines',opacity=0.25,name='+2'+ "\u03C3"))
    tc_legend = Image.open('Utilities/schematics/Furniture_Modeling_TC.jpg')
    fig.add_layout_image(source = tc_legend, xref = 'x domain', yref = 'y domain', xanchor = 'right', yanchor = 'top', x = 0.95, y = 0.95, sizex = 0.4, sizey = 0.5)

    return()

def plot_plume_vel_data(df):

    DP_dict = {'DPPlume1': 'black', 'DPPlume2': 'blue', 'DPPlume3': 'red'}

    for col in df.columns:
        if 'mean' in col:
            DP_str = col.split('_mean')[0]
            col_str = col.replace('_','/')
            col_std = col.replace('mean', 'std')
            fig.add_trace(go.Scatter(x=df.index, y=df[col], marker=dict(color=DP_dict[DP_str], size=8),name=DP_str))
            y_upper = df[col] + 2*df[col_std]
            y_lower = df[col] - 2*df[col_std]
            fig.add_trace(go.Scatter(x=y_lower.index,y=y_lower,fill=None, mode='lines', line_color= DP_dict[DP_str], hoveron='points',name='-2'+ "\u03C3"))
            fig.add_trace(go.Scatter(x=y_upper.index,y=y_upper,fill='tonexty',hoveron='points',line_color=DP_dict[DP_str],mode='lines',opacity=0.25,name='+2'+ "\u03C3"))
    bdp_legend = Image.open('Utilities/schematics/Furniture_Modeling_BDP.jpg')
    fig.add_layout_image(source = bdp_legend, xref = 'x domain', yref = 'y domain', xanchor = 'right', yanchor = 'top', x = 0.95, y = 0.95, sizex = 0.4, sizey = 0.5)

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

def format_and_save_plot(quantity, file_loc):

    label_dict = {'HF': 'Heat Flux (kW/m<sup>2</sup>)', 'Temp': 'Centerline Plume Temperature (\u00B0C)', 'Vel': 'Centerline Plume Velocity (m/s)', 'Heat Release Rate': 'HRR (kW)', 'Convective HRR': 'HRR (kW)', 'MLR': 'Mass Loss Rate (kg/s)', 'Load Cell':'Mass (kg)' , 'Total Heat Released': 'Total Heat Released (MJ)', 'SEA': 'Specific Extinction Area', 'Extinction Coefficient': 'Extinction Coefficient (1/m)', 'CO': 'CO Yield (g/g)', 'Soot': 'Soot Yield (g/g)'}

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

    # fig.add_layout_image(source = 'Utilities/schematics/Furniture_Modeling_HF.jpg', xref = 'x domain', yref = 'y domain', x = 1, y = 1, xanchor = 'right', yanchor = 'top', sizex = 0.2, sizey = 0.35)

    fig.write_html(file_loc.replace(' ','_'),include_plotlyjs="cdn")
    plt.close()

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

quant_list = ['Heat Release Rate', 'Convective HRR', 'Total Heat Released', 'Load Cell', 'MLR']

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    df_dict = {}
    material = d
    summary_df = pd.DataFrame()
    output_df = pd.DataFrame()
    co_df = pd.DataFrame()
    soot_df = pd.DataFrame()
    if os.path.isdir(f'{data_dir}{d}/Furniture_Calorimeter/'):
        print(material + ' Furniture Calorimeter')
        data_df = pd.DataFrame()
        reduced_df = pd.DataFrame()
        for f in sorted(glob.iglob(f'{data_dir}{d}/Furniture_Calorimeter/*.csv')):
            label = f.split('.csv')[0]

            data_temp_df = pd.read_csv(f, header = 0, index_col = 'Time (s)')
            
            data_temp_df['MLR_grad'] = -np.gradient(data_temp_df['Load Cell'], 1)
            data_temp_df['MLR'] = apply_savgol_filter(data_temp_df['MLR_grad'])

            df_dict[label] = data_temp_df.copy()

            # summary_df.at['Peak HRR (kW)', label] = float("{:.2f}".format(max(data_temp_df['Heat Release Rate'])))
            # summary_df.at['Time to Peak HRR (s)', label] = data_temp_df['Heat Release Rate'].idxmax()

            # summary_df.at['Total Heat Released (MJ)', label] = float("{:.2f}".format(data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'THR']))
            # total_mass_lost = data_temp_df.at['1','Sample Mass'] - data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'Sample Mass']
            # summary_df.at['Avg. Effective Heat of Combustion (MJ/kg)', label] = float("{:.2f}".format(((data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'THR'])*surf_area_m2)/(total_mass_lost/1000)))
            # summary_df.at['Initial Mass (g)', label] = scalar_data_series.at['SPECIMEN MASS']
            # summary_df.at['Final Mass (g)', label] = float("{:.2f}".format(data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'Sample Mass'] - holder_mass))

            # summary_df.at['Avg. Mass Loss Rate [10% to 90%] (g/m2s)', label] = float("{:.2f}".format(np.mean(data_temp_df.loc[t10:t90,'MLR']/surf_area_m2)))

            plot_dir = f'../03_Charts/{material}/Furniture_Calorimeter/'
            rep_str = label.split('_')[-1]

            # fig = go.Figure()
            # plot_hf_data(data_temp_df)
            # format_and_save_plot('HF', f'{plot_dir}{material}_{rep_str}_HF.html')

            # fig = go.Figure()
            # plot_plume_temp_data(data_temp_df)
            # format_and_save_plot('Temp', f'{plot_dir}{material}_{rep_str}_Temp.html')

            # fig = go.Figure()
            # plot_plume_vel_data(data_temp_df)
            # format_and_save_plot('Vel', f'{plot_dir}{material}_{rep_str}_Vel.html')

        for n in quant_list:
            fig = go.Figure()
            for key, value in df_dict.items():
                rep_str = key.split('_')[-1]
                plot_df = df_dict[key].filter(regex = n)
                try:
                    plot_data(plot_df, rep_str)
                except:
                    continue
            plot_dir = f'../03_Charts/{material}/Furniture_Calorimeter/'

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            format_and_save_plot(n, f'{plot_dir}{material}_{n}.html')

        # Plot heat flux data

        fig = go.Figure()
        plot_hf_temp = pd.DataFrame()
        plot_hf_df = pd.DataFrame()
        hfg_list = ['HF1', 'HF2', 'HF3', 'HF4', 'HF5', 'HF6']
        for hfg in hfg_list:
            for key, value in df_dict.items():
                rep_str = key.split('_')[-1]
                plot_hf_temp[f'{hfg}_{rep_str}'] = df_dict[key][hfg]
            plot_hf_df[f'{hfg}_mean'] = plot_hf_temp.filter(regex = hfg).dropna(axis = 'index').mean(axis=1)
            plot_hf_df[f'{hfg}_std'] = plot_hf_temp.filter(regex = hfg).dropna(axis = 'index').std(axis=1)

        plot_hf_data(plot_hf_df.loc[0:, :])

        plot_dir = f'../03_Charts/{material}/Furniture_Calorimeter/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        format_and_save_plot('HF', f'{plot_dir}{material}_HF.html')

        # Plot plume temperature data

        fig = go.Figure()
        plot_tc_temp = pd.DataFrame()
        plot_tc_df = pd.DataFrame()
        tc_list = ['TCPlume1', 'TCPlume2', 'TCPlume3']
        for tc in tc_list:
            for key, value in df_dict.items():
                rep_str = key.split('_')[-1]
                plot_tc_temp[f'{tc}_{rep_str}'] = df_dict[key][tc]
            plot_tc_df[f'{tc}_mean'] = plot_tc_temp.filter(regex = tc).dropna(axis = 'index').mean(axis=1)
            plot_tc_df[f'{tc}_std'] = plot_tc_temp.filter(regex = tc).dropna(axis = 'index').std(axis=1)

        plot_plume_temp_data(plot_tc_df.loc[0:, :])

        plot_dir = f'../03_Charts/{material}/Furniture_Calorimeter/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        format_and_save_plot('Temp', f'{plot_dir}{material}_TCPlume.html')

        # Plot plume velocity data

        fig = go.Figure()
        plot_dp_temp = pd.DataFrame()
        plot_dp_df = pd.DataFrame()
        dp_list = ['DPPlume1', 'DPPlume2', 'DPPlume3']
        for dp in dp_list:
            for key, value in df_dict.items():
                rep_str = key.split('_')[-1]
                plot_dp_temp[f'{dp}_{rep_str}'] = apply_savgol_filter(df_dict[key][dp])
            plot_dp_df[f'{dp}_mean'] = plot_dp_temp.filter(regex = dp).dropna(axis = 'index').mean(axis=1)
            plot_dp_df[f'{dp}_std'] = plot_dp_temp.filter(regex = dp).dropna(axis = 'index').std(axis=1)

        plot_plume_vel_data(plot_dp_df.loc[0:, :])

        plot_dir = f'../03_Charts/{material}/Furniture_Calorimeter/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        format_and_save_plot('Vel', f'{plot_dir}{material}_Plume_Vel.html')

    else:
        continue

    # summary_df = summary_df.round(1)
    # summary_df.sort_index(axis=1, inplace=True)
    # summary_df.to_csv(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_Data.csv', float_format='%.1f')

    # co_html_df = pd.DataFrame(index = hf_list, columns = ['Mean CO Yield [g/g]', 'CO Yield Std. Dev. [g/g]'])
    # hoc_html_df = pd.DataFrame(index = hf_list, columns = ['Mean Effective Heat of Combustion [MJ/kg]', 'Effective Heat of Combustion Std. Dev. [MJ/kg]'])

    # for hf in hf_list:
    #     html_df = output_df.filter(like=hf)
    #     html_df = html_df.rename(columns=lambda x: hf +' kW/m\u00b2 ' + x.split('_')[-1])
    #     html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_HRRPUA_Table_{hf}.html', float_format='%.2f', encoding='UTF-8', border=0)

    #     co_hf_df = co_df.filter(like=hf)
    #     co_html_df.loc[hf,'Mean CO Yield [g/g]'] = np.around(co_hf_df.mean(axis=1).to_numpy()[0], decimals=3)
    #     co_html_df.loc[hf, 'CO Yield Std. Dev. [g/g]'] = np.around(co_hf_df.std(axis=1).to_numpy()[0], decimals=3)
    #     co_html_df.index.rename('Incident Heat Flux [kW/m\u00b2]',inplace=True)

    #     soot_hf_df = soot_df.filter(like=hf)
    #     soot_hf_df = soot_hf_df.drop(columns=[col for col in soot_hf_df.columns if soot_hf_df[col].lt(0).any()])
    #     soot_html_df.loc[hf,'Mean Soot Yield [g/g]'] = np.around(soot_hf_df.mean(axis=1).to_numpy()[0], decimals=3)
    #     soot_html_df.loc[hf, 'Soot Yield Std. Dev. [g/g]'] = np.around(soot_hf_df.std(axis=1).to_numpy()[0], decimals=3)
    #     soot_html_df.index.names = ['Incident Heat Flux [kW/m\u00b2]']

    #     hoc_df = summary_df.filter(like=hf)
    #     hoc_df = hoc_df.filter(regex = 'Heat of Combustion', axis = 'index')
    #     hoc_html_df.loc[hf, 'Mean Effective Heat of Combustion [MJ/kg]'] = np.around(hoc_df.mean(axis=1).to_numpy()[0], decimals=1)
    #     hoc_html_df.loc[hf, 'Effective Heat of Combustion Std. Dev. [MJ/kg]'] = np.around(hoc_df.std(axis=1).to_numpy()[0], decimals=1)
    #     hoc_html_df.index.names = ['Incident Heat Flux [kW/m\u00b2]']

    # co_html_df = co_html_df.reset_index()
    # co_html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_CO_Table.html',index=False, encoding='UTF-8', border=0)

    # if soot_html_df.isnull().values.any():
    #     pass
    # else:
    #     soot_html_df = soot_html_df.reset_index()
    #     soot_html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_Soot_Table.html',index=False, encoding='UTF-8', border=0)

    # hoc_html_df = hoc_html_df.reset_index()
    # hoc_html_df.to_html(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_EHC_Table.html',index=False, encoding='UTF-8', border=0)