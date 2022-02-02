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
from tkinter import Tk
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import git

# Define variables #
data_dir = '../01_Data/'
save_dir = '../03_Charts/'

hf_list = ['25', '50', '75']
quant_list = ['HRRPUA', 'MLR', 'SPR', 'SEA', 'Extinction Coefficient', 'EHC']

y_max_dict = {'HRRPUA':500, 'MLR':1, 'SPR':5, 'SEA':1000, 'Extinction Coefficient':2, 'EHC':50}
y_inc_dict = {'HRRPUA':100, 'MLR':0.2, 'SPR':1, 'SEA':200, 'Extinction Coefficient':0.5, 'EHC':10}

output_df = pd.DataFrame()

equal_scales = False

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

def create_1plot_fig():
	# Define figure for the plot
	fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
	#plt.subplots_adjust(left=0.08, bottom=0.3, right=0.92, top=0.95)

	# Reset values for x & y limits
	x_min, x_max, y_min, y_max = 0, 0, 0, 0

	return(fig, ax1, x_min, x_max, y_min, y_max)

def plot_data(df, rep):

	rep_dict = {'R1': 'k', 'R2': 'b', 'R3': 'r', 'R4': 'g'}

	ax1.plot(df.index, df.iloc[:,0], color=rep_dict[rep], ls='-', marker=None, label = rep)

	y_max = max(df.iloc[:,0])
	y_min = min(df.iloc[:,0])

	x_max = max(df.index)
	x_min = min(df.index)
	
	return(y_min, y_max, x_min, x_max)

def air_density(temperature):
	# returns density in kg/m3 given a temperature in C
	rho = 1.2883 - 4.327e-3*temperature + 8.78e-6*temperature**2
	return rho

def format_and_save_plot(xlims, ylims, inc, quantity, file_loc):
	
	label_dict = {'HRRPUA': 'Heat Release Rate (kW/m2)', 'MLR': 'Mass Loss Rate (g/s)', 'EHC':'Effective Heat of Combustion (MJ/kg)' , 'SPR': 'Smoke Production Rate (1/s)', 'SEA': 'Specific Extinction Area', 'Extinction Coefficient': 'Extinction Coefficient (1/m)'}

	# Set tick parameters
	ax1.tick_params(labelsize=tick_size, length=8, width=0.75, direction = 'inout')

	# Scale axes limits & labels
	ax1.set_ylim(bottom=ylims[0], top=ylims[1])
	ax1.set_xlim(left=xlims[0], right=xlims[1])
	ax1.set_xlabel('Time (s)', fontsize=label_size)

	# ax1.set_position([0.15, 0.3, 0.77, 0.65])

	y_range_array = np.arange(ylims[0], ylims[1] + inc, inc)
	ax1.set_ylabel(label_dict[quantity], fontsize=label_size)

	yticks_list = list(y_range_array)

	x_range_array = np.arange(xlims[0], xlims[1] + 120, 120)
	xticks_list = list(x_range_array)

	if quantity == 'HRRPUA':
		ax1.autoscale(enable = True, axis = 'both')
	else:
		ax1.set_yticks(yticks_list)
		ax1.autoscale(enable = True, axis = 'x')

	ax1.tick_params(axis = 'x', labelrotation = 45)

	# ax2 = ax1.secondary_yaxis('right')
	ax2 = ax1.twinx()
	ax2.tick_params(axis='y', direction='in', length = 4)
	ax2.set_yticks(yticks_list)
	empty_labels = ['']*len(yticks_list)
	ax2.set_yticklabels(empty_labels)

	# ax3 = ax1.secondary_xaxis('top')
	ax3 = ax1.twiny()
	ax3.tick_params(axis='x', direction='in', length = 4)
	ax3.set_xticks(xticks_list)
	empty_labels = ['']*len(xticks_list)
	ax3.set_xticklabels(empty_labels)

	#Get github hash to display on graph
	repo = git.Repo(search_parent_directories=True)
	sha = repo.head.commit.hexsha
	short_sha = repo.git.rev_parse(sha, short=True)
	# short_sha = '*** need python git pkg ***'

	ax1.text(1, 1,'Repository Version: ' + short_sha,
		horizontalalignment='right',
		verticalalignment='bottom',
		transform = ax1.transAxes)

	# Add legend
	handles1, labels1 = ax1.get_legend_handles_labels()

	# print(f'handles1: {handles1}')

	# # order = []
	# # order.append(labels1.index('Wet Sample Preparation'))
	# # order.append(labels1.index('Dry Sample Preparation'))

	# handles1 = handles1[i]
	# labels1 = labels1[i]

	# n_col, leg_list, leg_labels = legend_entries(handles1, labels1)

	#a_list = [a_list[i] for i in order]

	plt.legend(handles1, labels1, loc = 'upper center', bbox_to_anchor = (0.5, -0.27), fontsize=16,
				handlelength=2, frameon=True, framealpha=1.0, ncol=3)

	# Clean up whitespace padding
	#fig.tight_layout()

	# Save plot to file
	plt.savefig(file_loc)
	plt.close()

	print('Plotting ' + label_dict[quantity])

for d in os.scandir(data_dir):
	df_dict = {}
	material = d.path.split('/')[-1]
	if material == '.DS_Store':
		continue
	
	### CHOOSE MATERIAL ###
	if material != 'LDPE':
		continue
	#######################

	plot_data_df = pd.DataFrame()
	print(f'{material} Cone')
	if d.is_dir():
		if os.path.isdir(f'{d.path}/Cone/'):
			data_df = pd.DataFrame()
			reduced_df = pd.DataFrame()
			for f in glob.iglob(f'{d.path}/Cone/*.csv'):
			# for f in os.scandir(f'{d.path}/Cone/'):
				if 'scalar' in f.lower() or 'cone_analysis_data' in f.lower():
					continue
				else:
					label_list = f.split('.csv')[0].split('_')
					label = label_list[-3].split('Scan')[0] + '_' + label_list[-1]
					data_temp_df = pd.read_csv(f, header = 0, skiprows = [1, 2, 3, 4], index_col = 'Names')

					scalar_data_fid = f.replace('Scan','Scalar')
					scalar_data_series = pd.read_csv(scalar_data_fid, index_col = 0, squeeze='True')

					c_factor = float(scalar_data_series.at['C FACTOR'])

					data_temp_df['O2 Meter'] = data_temp_df['O2 Meter']/100
					data_temp_df['CO2 Meter'] = data_temp_df['CO2 Meter']/100
					data_temp_df['CO Meter'] = data_temp_df['CO Meter']/100

					data_temp_df.loc[:,'EDF'] = ((data_temp_df.loc[:,'Exh Press']/(data_temp_df.loc[:,'Stack TC']+273.15)).apply(np.sqrt)).multiply(c_factor) # Exhaust Duct Flow (m_e_dot)
					data_temp_df.loc[:,'Volumetric Flow'] = data_temp_df.loc[:,'EDF']*air_density(data_temp_df.loc[:,'Smoke TC']) # Exhaust Duct Flow (m_e_dot)
					# O2_offset = 0.2095 - data_temp_df.at['Baseline', 'O2 Meter']
					# data_temp_df.loc[:,'ODF'] = (0.2095 - data_temp_df.loc[:,'O2 Meter'] + O2_offset) / (1.105 - (1.5*(data_temp_df.loc[:,'O2 Meter'] + O2_offset))) # Oxygen depletion factor with only O2
					data_temp_df.loc[:,'ODF'] = (data_temp_df.at['Baseline', 'O2 Meter'] - data_temp_df.loc[:,'O2 Meter']) / (1.105 - (1.5*(data_temp_df.loc[:,'O2 Meter']))) # Oxygen depletion factor with only O2                    
					data_temp_df.loc[:,'ODF_ext'] = (data_temp_df.at['Baseline', 'O2 Meter']*(1-data_temp_df.loc[:, 'CO2 Meter'] - data_temp_df.loc[:, 'CO Meter']) - data_temp_df.loc[:, 'O2 Meter']*(1-data_temp_df.at['Baseline', 'CO2 Meter']))/(data_temp_df.at['Baseline', 'O2 Meter']*(1-data_temp_df.loc[:, 'CO2 Meter']-data_temp_df.loc[:, 'CO Meter']-data_temp_df.loc[:, 'O2 Meter'])) # Oxygen Depletion Factor with O2, CO, and CO2
					data_temp_df.loc[:,'HRR'] = 1.10*(e)*data_temp_df.loc[:,'EDF']*data_temp_df.loc[:,'ODF']
					data_temp_df.loc[:,'HRR_ext'] = 1.10*(e)*data_temp_df.loc[:,'EDF']*data_temp_df.at['Baseline', 'O2 Meter']*((data_temp_df.loc[:,'ODF_ext']-0.172*(1-data_temp_df.loc[:,'ODF'])*(data_temp_df.loc[:, 'CO2 Meter']/data_temp_df.loc[:, 'O2 Meter']))/((1-data_temp_df.loc[:,'ODF'])+1.105*data_temp_df.loc[:,'ODF']))
					data_temp_df.loc[:,'HRRPUA'] = data_temp_df.loc[:,'HRR']/float(scalar_data_series.at['SURF AREA'])
					data_temp_df['THR'] = 0.25*data_temp_df['HRRPUA'].cumsum()/1000
					data_temp_df['MLR_grad'] = -np.gradient(data_temp_df['Sample Mass'], 0.25)
					data_temp_df['MLR'] = apply_savgol_filter(data_temp_df['MLR_grad'])
					data_temp_df['MLR'][data_temp_df['MLR'] > 5] = 0
					
					# # MLR Calculation
					# data_temp_df['MLR'] = np.zeros(len(data_temp_df['Sample Mass']))
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
					data_temp_df['Extinction Coefficient'] = data_temp_df['Ext Coeff'] - data_temp_df.at['Baseline','Ext Coeff']
					data_temp_df['SPR'] = (data_temp_df.loc[:,'Extinction Coefficient'] * data_temp_df.loc[:,'Volumetric Flow'])/float(scalar_data_series.at['SURF AREA'])
					data_temp_df['SPR'][data_temp_df['SPR'] < 0] = 0
					data_temp_df['SEA'] = (1000*data_temp_df.loc[:,'Volumetric Flow']*data_temp_df.loc[:,'Extinction Coefficient'])/data_temp_df['MLR']
					# data_temp_df['SEA'][np.isinf(data_temp_df['SEA'])] = np.nan

					df_dict[label] = data_temp_df[['Time', 'HRRPUA', 'MLR', 'EHC', 'SPR', 'SEA', 'Extinction Coefficient']].copy()
					df_dict[label].set_index(df_dict[label].loc[:,'Time'], inplace = True)
					df_dict[label] = df_dict[label][df_dict[label].index.notnull()]
					df_dict[label].drop('Time', axis = 1, inplace = True)
					end_time = float(scalar_data_series.at['END OF TEST TIME'])
					num_intervals = (max(df_dict[label].index)-end_time)/0.25
					drop_list = list(np.linspace(end_time, max(df_dict[label].index), int(num_intervals+1)))
					df_dict[label].drop(labels = drop_list, axis = 0, inplace = True)

					output_df.at['Time to Sustained Ignition (s)', label] = float(scalar_data_series.at['TIME TO IGN'])
					output_df.at['Peak HRRPUA (kW/m2)', label] = max(data_temp_df['HRRPUA'])
					output_df.at['Time to Peak HRRPUA (s)', label] = data_temp_df.loc[data_temp_df['HRRPUA'].idxmax(), 'Time'] - float(scalar_data_series.at['TIME TO IGN'])
					ign_index = data_temp_df.index[data_temp_df['Time'] == float(scalar_data_series.at['TIME TO IGN'])][0]
					t60 = str(int(ign_index) + 240)
					t180 = str(int(ign_index) + 720)
					t300 = str(int(ign_index) + 1200)

					try: output_df.at['Average HRRPUA over 60 seconds (kW/m2)', label] = np.mean(data_temp_df.loc[ign_index:t60,'HRRPUA'])
					except: output_df.at['Average HRRPUA over 60 seconds (kW/m2)', label] = math.nan

					try: output_df.at['Average HRRPUA over 180 seconds (kW/m2)', label] = np.mean(data_temp_df.loc[ign_index:t180,'HRRPUA'])
					except: output_df.at['Average HRRPUA over 180 seconds (kW/m2)', label] = math.nan
					
					try: output_df.at['Average HRRPUA over 300 seconds (kW/m2)', label] = np.mean(data_temp_df.loc[ign_index:t300,'HRRPUA'])
					except: output_df.at['Average HRRPUA over 300 seconds (kW/m2)', label] = math.nan

					output_df.at['Total Heat Released (MJ/m2)', label] = data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'THR']
					total_mass_lost = data_temp_df.at['1','Sample Mass'] - data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'Sample Mass']
					holder_mass = data_temp_df.at['1','Sample Mass'] - float(scalar_data_series.at['SPECIMEN MASS'])
					output_df.at['Avg. Effective Heat of Combustion (MJ/kg)', label] = ((data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'THR'])*float(scalar_data_series.at['SURF AREA']))/(total_mass_lost/1000)
					output_df.at['Initial Mass (g)', label] = scalar_data_series.at['SPECIMEN MASS']
					output_df.at['Final Mass (g)', label] = data_temp_df.at[scalar_data_series.at['END OF TEST SCAN'],'Sample Mass'] - holder_mass
					output_df.at['Mass at Ignition (g)', label] = data_temp_df.at[ign_index,'Sample Mass'] - holder_mass
					
					t10 = data_temp_df['Sample Mass'].sub(data_temp_df.at['1','Sample Mass'] - 0.1*total_mass_lost).abs().idxmin()
					t90 = data_temp_df['Sample Mass'].sub(data_temp_df.at['1','Sample Mass'] - 0.9*total_mass_lost).abs().idxmin()

					output_df.at['Avg. Mass Loss Rate [10% to 90%] (g/m2s)', label] = np.mean(data_temp_df.loc[t10:t90,'MLR']/float(scalar_data_series.at['SURF AREA']))                    
					
			for n in quant_list:
				for m in hf_list:
					ylims = [0,0]
					xlims = [0,0]
					fig, ax1, x_min, x_max, y_min, y_max = create_1plot_fig()
					for key, value in df_dict.items():
						rep_str = key.split('_')[-1]
						if m in key:
							plot_df = df_dict[key].filter(regex = n)
							ymin, ymax, xmin, xmax = plot_data(plot_df, rep_str)
					
					y_min = max(ymin, y_min)
					x_min = max(xmin, x_min)
					y_max = max(ymax, y_max)
					x_max = max(xmax, x_max)

					inc = y_inc_dict[n]

					if equal_scales or np.isinf(y_max) or np.isinf(y_min):
						ylims[0] = 0
						ylims[1] = y_max_dict[n]
						xlims[0] = 0
						xlims[1] = 120 * (math.ceil(x_max/120))
					else:
						ylims[0] = y_min - abs(y_min * 0.1)
						ylims[1] = y_max * 1.1
						xlims[0] = x_min
						xlims[1] = x_max

					plot_dir = f'../03_Charts/{material}/Cone/'

					if not os.path.exists(plot_dir):
						os.makedirs(plot_dir)

					format_and_save_plot(xlims, ylims, inc, n, f'{plot_dir}{material}_Cone_{n}_{m}.pdf')       

		else:
			continue
	else:
		continue

	output_df.sort_index(axis=1, inplace=True)
	output_df.to_csv(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_Data.csv', float_format='%.2f')