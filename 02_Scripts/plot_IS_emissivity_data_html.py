# Calculate emissivity from FTIR integrating sphere
#   by: ULRI's Fire Safety Research Institute
#   Questions? Submit them here: https://github.com/ulfsri/fsri_materials_database/issues

# ***************************** Usage Notes *************************** #
# - Script outputs as a function of source temperature                  #
#   -  HTML Graphs: dir: /03_Charts/{Material}/FTIR/IS                  #
#      Graphs: Panel Emissivity                                         #
#                                                                       #
#      HTML Tables dir: /01_Data/{Material}/FTIR/IS                     #
#      Tables: Panel Emissivity                                         #
# ********************************************************************* #

import glob
import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import math
import matplotlib.pyplot as plt
import re
from scipy import integrate
import plotly.graph_objects as go
import git

h = 6.62607015e-34 # J-s (Planck's Constant)
c = 299702547 # m/s (speed of light in air)
kB = 1.380649e-23 # J/K (Boltzmann constant)
sig = 5.67e-8 # W/m2K4

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

def plot_mean_data(df):
	fig.add_trace(go.Scatter(x=df.index, y=df['Emissivity'],error_y=dict(type='data', array=2*df['Std. Dev.']),mode='markers',name='Mean', marker=dict(color='blue', size=8)))

	return()

def reindex_data(data, min_x, max_x):
	data = data.reindex(data.index.union(np.arange(min_x, max_x+1, 1)))
	data = data.astype('float64')
	data.index = data.index.astype('float64')
	data = data.interpolate(method='cubic')

	data = data.loc[np.arange(min_x, max_x+1, 1.0)]

	return data

def format_and_save_plot(file_loc,material):
	fig.update_layout(yaxis_title='Total Hemispherical Emissivity')

	fig.update_layout(autosize=False, width=513, height=450,margin=dict(l=25,r=25,b=40,t=40,pad=4))

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

	# Save plot to file
	fig.update_layout(xaxis_title='Source Temperature [K]', font=dict(size=18))
	fig.write_html(file_loc,include_plotlyjs="cdn")
	plt.close()

	# print()

t_range = np.linspace(600,2000,8)

for material in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
	sample_df = pd.DataFrame()
	trap_df = pd.DataFrame()
	if os.path.isdir(f'{data_dir}{material}/FTIR/IS'):
		print(material + ' FTIR IS')
		for f in sorted(glob.iglob(f'{data_dir}{material}/FTIR/IS/*.dpt')):
			if '_S_' in f:
				rep = f.split('.dpt')[0].split('_')[-1]
				temp_df = pd.read_csv(f, index_col = 0, header=None, names=['wavenumber', 'reflectivity'])
				min_x = math.ceil(min(temp_df.index))
				max_x = math.floor(max(temp_df.index))
				temp_df = reindex_data(temp_df, min_x, max_x)

				if sample_df.empty:
					sample_df = temp_df
					sample_df.rename(columns = {'reflectivity':rep}, inplace=True)
				else:
					sample_df[rep] = temp_df
			elif '_T_' in f:
				rep = f.split('.dpt')[0].split('_')[-1]
				temp_df = pd.read_csv(f, index_col = 0, header=None, names=['wavenumber', 'reflectivity'])
				min_x = math.ceil(min(temp_df.index))
				max_x = math.floor(max(temp_df.index))
				temp_df = reindex_data(temp_df, min_x, max_x)

				if trap_df.empty:
					trap_df = temp_df
					trap_df.rename(columns = {'reflectivity':rep}, inplace=True)
				else:
					trap_df[rep] = temp_df
			else:
				continue

		if sample_df.empty or trap_df.empty:
			continue
		else:
			sample_df = sample_df.rename_axis('wavenumber')
			trap_df = trap_df.rename_axis('wavenumber')
			sample_df['Mean'] = sample_df.mean(axis=1)
			sample_df['Std'] = sample_df.std(axis=1)
			trap_df['Mean'] = sample_df.mean(axis=1)
			trap_df['Std'] = sample_df.std(axis=1)

			mean_reflect = pd.DataFrame(index = sample_df.index)

			mean_reflect['Sample'] = sample_df['Mean']
			mean_reflect['Sample_std'] = sample_df['Std']
			mean_reflect['LT'] = trap_df['Mean']
			mean_reflect['LT_std'] = sample_df['Std']

			mean_reflect['wavelength'] = 10000000/mean_reflect.index # wavelength in nm
			mean_reflect.set_index('wavelength', inplace=True)

			for ind in mean_reflect.index:
				i = ind/1000000000 # convert from nm to m
				mean_reflect.loc[ind, 'measured'] = 1-((mean_reflect.loc[ind, 'Sample']-mean_reflect.loc[ind, 'LT'])/(1-mean_reflect.loc[ind, 'LT']))
				# uncertainty propagation for measured reflectance
				mean_reflect.loc[ind, 'measured_std_comp_1'] = np.sqrt(mean_reflect.loc[ind, 'Sample_std']**2+mean_reflect.loc[ind, 'LT_std']**2)
				mean_reflect.loc[ind, 'measured_std_comp_1_rel'] = mean_reflect.loc[ind, 'measured_std_comp_1']/(mean_reflect.loc[ind, 'Sample']-mean_reflect.loc[ind, 'LT'])
				mean_reflect.loc[ind, 'measured_std_comp_2_rel'] = mean_reflect.loc[ind, 'LT_std']/mean_reflect.loc[ind, 'LT']
				mean_reflect.loc[ind, 'measured_std'] = ((mean_reflect.loc[ind, 'Sample']-mean_reflect.loc[ind, 'LT'])/(1-mean_reflect.loc[ind, 'LT']))*np.sqrt(mean_reflect.loc[ind, 'measured_std_comp_1_rel']**2 + mean_reflect.loc[ind, 'measured_std_comp_2_rel']**2)
				for t_source in t_range:
					mean_reflect.loc[ind, str(t_source) +'_spectrum'] = (2*h*c**2)/(i**5*(math.exp((h*c)/(i*kB*t_source))-1)) # Planck's Law [W/m^3]
					mean_reflect.loc[ind, str(t_source) +'_measured'] = (2*h*c**2*mean_reflect.loc[ind, 'measured'])/(i**5*(math.exp((h*c)/(i*kB*t_source))-1))
					mean_reflect.loc[ind, str(t_source) +'_measured_std'] = (2*h*c**2*mean_reflect.loc[ind, 'measured_std'])/(i**5*(math.exp((h*c)/(i*kB*t_source))-1))

			ref_data = pd.DataFrame(index = t_range, columns = ['Emissivity', 'Std. Dev.'])

			for t_source in t_range:
				ref_data.at[t_source, 'Emissivity'] = integrate.trapz(mean_reflect[str(t_source) +'_measured'], x=mean_reflect.index)/integrate.trapz(mean_reflect[str(t_source) +'_spectrum'], x=mean_reflect.index)
				ref_data.loc[t_source, 'Std. Dev.'] = integrate.trapz(mean_reflect[str(t_source) +'_measured_std'], x=mean_reflect.index)/integrate.trapz(mean_reflect[str(t_source) +'_spectrum'], x=mean_reflect.index)

			# ref_data = pd.DataFrame(ref_data).reset_index()
			ref_data = ref_data.astype('float64')
			ref_data['Emissivity'] = ref_data['Emissivity'].round(3)
			ref_data['Std. Dev.'] = ref_data['Std. Dev.'].round(3)

			# print(ref_data)

			ref_data.index.rename('Source Temperature [K]', inplace=True)
			ref_data = ref_data.reset_index()
			ref_data.to_html(f'{data_dir}{material}/FTIR/IS/{material}_Emissivity.html', index=False,border=0)

			fig = go.Figure()
			plot_mean_data(ref_data)

			plot_dir = f'../03_Charts/{material}/FTIR/IS/'

			if not os.path.exists(plot_dir):
				os.makedirs(plot_dir)

			# print('Plotting Chart')
			format_and_save_plot(f'{plot_dir}{material}_Emissivity.html',material)