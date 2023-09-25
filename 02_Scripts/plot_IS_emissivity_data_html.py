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

plot_all = False
if not plot_all: 
    print('plot_all is set to False, so any materials with existing html output files will be skipped')

h = 6.62607015e-34 # J-s (Planck's Constant)
c = 299702547 # m/s (speed of light in air)
kB = 1.380649e-23 # J/K (Boltzmann constant)
sig = 5.67e-8 # W/m2K4

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

def plot_mean_data(df):
	fig.add_trace(go.Scatter(x=df.index, y=df['Emissivity'],error_y=dict(type='data', array=2*df['Std. Dev.']),mode='markers',name='Mean', marker=dict(color='blue', size=8)))

	return()

def reindex_data(data, min_x, max_x, dx):
	data = data.reindex(data.index.union(np.arange(min_x, max_x+dx, dx)))
	data = data.astype('float64')
	data.index = data.index.astype('float64')
	data = data.interpolate(method='cubic')

	data = data.loc[np.arange(min_x, max_x+dx, dx)]

	return data

def format_and_save_plot(file_loc,material):
	fig.update_layout(yaxis_title='Total Hemispherical Emissivity')

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

	# Save plot to file
	fig.update_layout(xaxis_title='Source Temperature [K]', font=dict(size=18))
	fig.write_html(file_loc,include_plotlyjs="cdn")
	plt.close()

	# print()

def lsqfity(X, Y):
	"""
	Calculate a "MODEL-1" least squares fit.

	The line is fit by MINIMIZING the residuals in Y only.

	The equation of the line is:     Y = my * X + by.

	Equations are from Bevington & Robinson (1992)
	Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
	pp: 104, 108-109, 199.

	Data are input and output as follows:

	my, by, ry, smy, sby = lsqfity(X,Y)
	X     =    x data (vector)
	Y     =    y data (vector)
	my    =    slope
	by    =    y-intercept
	ry    =    correlation coefficient
	smy   =    standard deviation of the slope
	sby   =    standard deviation of the y-intercept

	"""

	X, Y = map(np.asanyarray, (X, Y))

	# Determine the size of the vector.
	n = len(X)

	# Calculate the sums.

	Sx = np.sum(X)
	Sy = np.sum(Y)
	Sx2 = np.sum(X ** 2)
	Sxy = np.sum(X * Y)
	Sy2 = np.sum(Y ** 2)

	# Calculate re-used expressions.
	num = n * Sxy - Sx * Sy
	den = n * Sx2 - Sx ** 2

	# Calculate my, by, ry, s2, smy and sby.
	my = num / den
	by = (Sx2 * Sy - Sx * Sxy) / den
	ry = num / (np.sqrt(den) * np.sqrt(n * Sy2 - Sy ** 2))

	diff = Y - by - my * X

	s2 = np.sum(diff * diff) / (n - 2)
	smy = np.sqrt(n * s2 / den)
	sby = np.sqrt(Sx2 * s2 / den)

	rmb = -Sx / np.sqrt(n * Sx2)
	smb = rmb * smy * sby

	return my, by, ry, smy, sby, s2, rmb, smb  

t_range = np.linspace(600,2000,8)

for material in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
	trans_sample_df = pd.DataFrame()
	trans_trap_df = pd.DataFrame()
	sample_df = pd.DataFrame()
	trap_df = pd.DataFrame()
	if os.path.isdir(f'{data_dir}{material}/FTIR/IS'):
		# if material != 'MDF': continue
		if not plot_all:
			if mat_status_df.loc[material, 'Emissivity']: 
				# print(f'Skipping {material} Cone --- plot_all is False and output charts exist')
				continue

		print(material + ' FTIR IS')
		fid_list = list(glob.iglob(f'{data_dir}{material}/FTIR/IS/*.dpt')) 
		fid_list_trans = [i for i in fid_list if 'Trans' in i]
		fid_list_ref = [i for i in fid_list if 'Trans' not in i]
		if fid_list_trans:
			thickness_list = []
			for f in sorted(fid_list_trans):
				rep = f.split('_')[-1].split('.dpt')[0]
				r = re.compile('\d\.\d*')
				thickness = list(filter(r.match, f.split('_')))[0]
				thickness_list.append(thickness)
				temp_df = pd.read_csv(f, index_col = 0, header=None, names=['wavenumber', 'transmittance'])
				min_x = math.ceil(min(temp_df.index))
				max_x = math.floor(max(temp_df.index))
				temp_df = reindex_data(temp_df, min_x, max_x, 1)
				if '_S_' in f:
					if trans_sample_df.empty:
						trans_sample_df = temp_df
						trans_sample_df.rename(columns = {'transmittance':f'{thickness}_{rep}'}, inplace=True)
					else:
						trans_sample_df[f'{thickness}_{rep}'] = temp_df
				elif '_T_' in f:
					if trans_trap_df.empty:
						trans_trap_df = temp_df
						trans_trap_df.rename(columns = {'transmittance':f'{thickness}_{rep}'}, inplace=True)
					else:
						trans_trap_df[f'{thickness}_{rep}'] = temp_df
				else:
					continue
			mean_trans = pd.DataFrame()
			th_list = []
			for d in sorted(set(thickness_list)):
				th_list.append(d)
				mean_trans[d] = trans_sample_df.filter(axis = 'columns', regex = d).mean(axis = 'columns') - trans_trap_df.filter(axis = 'columns', regex = d).mean(axis='columns') # correct for light trap background
			mean_trans['wavelength'] = 10000000/mean_trans.index
			mean_trans.set_index('wavelength', inplace=True)
			mean_trans = reindex_data(mean_trans, math.ceil(min(mean_trans.index)), math.floor(max(mean_trans.index))-10, 10)
			mean_trans['wl_m'] = mean_trans.index*1e-9
			trans_data = pd.DataFrame(index = t_range, columns = th_list)
			for t_source in t_range:
				mean_trans[f'{t_source}_spectrum'] = (2*h*c**2)/(mean_trans['wl_m']**5*(np.exp((h*c)/(mean_trans['wl_m']*kB*t_source))-1)) # Planck's Law [W/m^3]
				for y in th_list:
					mean_trans[f'{t_source}_measured_{y}'] = (2*h*c**2*mean_trans[y])/(mean_trans['wl_m']**5*(np.exp((h*c)/(mean_trans['wl_m']*kB*t_source))-1)) # Weighted spectral transmission
					# mean_trans[str(t_source) +'_measured_std'] = (2*h*c**2*mean_trans[f'{x}_std'])/(mean_trans['wl_m']**5*(math.exp((h*c)/(mean_trans['wl_m']*kB*t_source))-1))
					trans_data.at[t_source, y] = integrate.trapz(mean_trans[f'{t_source}_measured_{y}'], x=mean_trans.index)/integrate.trapz(mean_trans[f'{t_source}_spectrum'], x=mean_trans.index) # Mean total transmittance w.r.t. thickness Eq 6 in Linteris [10.1002/fam.1113]
					# ref_data.loc[t_source, 'Std. Dev.'] = integrate.trapz(mean_reflect[str(t_source) +'_measured_std'], x=mean_reflect.index)/integrate.trapz(mean_reflect[str(t_source) +'_spectrum'], x=mean_reflect.index)
			trans_data.loc['Mean', :] = trans_data.mean(axis='index')
			log_trans_data = trans_data.copy()
			for i in trans_data.columns:
				for t in trans_data.index:
					log_trans_data.at[t, i] = -np.log(np.absolute(trans_data.loc[t, i])) # -log of mean transmittance

		for f in sorted(fid_list_ref):
			if '_S_' in f:
				rep = f.split('.dpt')[0].split('_')[-1]
				temp_df = pd.read_csv(f, index_col = 0, header=None, names=['wavenumber', 'reflectivity'])
				min_x = math.ceil(min(temp_df.index))
				max_x = math.floor(max(temp_df.index))
				temp_df = reindex_data(temp_df, min_x, max_x, 1)

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
				temp_df = reindex_data(temp_df, min_x, max_x, 1)

				if trap_df.empty:
					trap_df = temp_df
					trap_df.rename(columns = {'reflectivity':rep}, inplace=True)
				else:
					trap_df[rep] = temp_df
			else:
				continue

		if sample_df.empty or trap_df.empty:
			print('\t-- skipping: empty data file')
			continue
		else:
			sample_df = sample_df.rename_axis('wavenumber')
			trap_df = trap_df.rename_axis('wavenumber')
			sample_df['Mean'] = sample_df.mean(axis=1)
			sample_df['Std'] = sample_df.std(axis=1)
			trap_df['Mean'] = trap_df.mean(axis=1)
			trap_df['Std'] = trap_df.std(axis=1)

			mean_reflect = pd.DataFrame(index = sample_df.index)

			mean_reflect['Sample'] = sample_df['Mean']
			mean_reflect['Sample_std'] = sample_df['Std']
			mean_reflect['LT'] = trap_df['Mean']
			mean_reflect['LT_std'] = trap_df['Std']

			mean_reflect['wavelength'] = 10000000/mean_reflect.index # wavelength in nm
			mean_reflect.set_index('wavelength', inplace=True)
			mean_reflect = reindex_data(mean_reflect, math.ceil(min(mean_reflect.index)), math.floor(max(mean_reflect.index))-10, 10)
			mean_reflect['wl_m'] = mean_reflect.index*1e-9 # convert from nm to m

			if fid_list_trans:
				d = max(thickness_list)
				mean_trans_d = pd.DataFrame(index = mean_trans.index, columns = ['transmittance'], data = np.zeros(len(mean_trans.index)))
				mean_trans_d['transmittance'] = mean_trans[d]

				abs_coeff = pd.DataFrame()
				thickness = sorted([float(i)/1000 for i in set(thickness_list)]) # thickness in m

				for t in log_trans_data.index:
					log_mean = list(log_trans_data.loc[t, :])
					my, by, ry, smy, sby, s2, rmb, smb = lsqfity(thickness, log_mean)

					abs_coeff.loc[t, 'Absorption Coefficient (1/m)'] = my
					abs_coeff.loc[t, 'Std. Dev.'] = smy # this does not account for uncertainty in measurement - only scatter in data

				abs_coeff['Source Temperature [K]'] = abs_coeff.index
				abs_coeff = abs_coeff[['Source Temperature [K]', 'Absorption Coefficient (1/m)', 'Std. Dev.']]
				abs_coeff = abs_coeff.round(1)
				abs_coeff.to_html(f'{data_dir}{material}/FTIR/IS/{material}_Absorption_Coefficient.html', index=False, border=0)

			else:
				mean_trans_d = pd.DataFrame(index = mean_reflect.index, columns = ['transmittance'], data = np.zeros(len(mean_reflect.index)))
			mean_trans_d = reindex_data(mean_trans_d, math.ceil(min(mean_reflect.index)), math.floor(max(mean_reflect.index)), 10)
			mean_reflect['transmittance'] = mean_trans_d['transmittance']

			mean_reflect['measured'] = 1-((mean_reflect['Sample']-mean_reflect['LT'])/(1-mean_reflect['LT'])) - mean_reflect['transmittance']

			# uncertainty propagation for measured reflectance
			mean_reflect['measured_std_comp_1'] = np.sqrt(mean_reflect['Sample_std']**2+mean_reflect['LT_std']**2)
			mean_reflect['measured_std_comp_1_rel'] = mean_reflect['measured_std_comp_1']/(mean_reflect['Sample']-mean_reflect['LT'])
			mean_reflect['measured_std_comp_2_rel'] = mean_reflect['LT_std']/mean_reflect['LT']
			mean_reflect['measured_std'] = ((mean_reflect['Sample']-mean_reflect['LT'])/(1-mean_reflect['LT']))*np.sqrt(mean_reflect['measured_std_comp_1_rel']**2 + mean_reflect['measured_std_comp_2_rel']**2)

			ref_data = pd.DataFrame(index = t_range, columns = ['Emissivity', 'Std. Dev.'])

			for ind in mean_reflect.index:
				i = ind*1e-9 # convert from nm to m
				for t_source in t_range:
					mean_reflect.loc[ind, str(t_source) +'_spectrum'] = (2*h*c**2)/(i**5*(math.exp((h*c)/(i*kB*t_source))-1)) # Planck's Law [W/m^3]
					mean_reflect.loc[ind, str(t_source) +'_measured'] = (2*h*c**2*mean_reflect.loc[ind, 'measured'])/(i**5*(math.exp((h*c)/(i*kB*t_source))-1))
					mean_reflect.loc[ind, str(t_source) +'_measured_std'] = (2*h*c**2*mean_reflect.loc[ind, 'measured_std'])/(i**5*(math.exp((h*c)/(i*kB*t_source))-1))

			mean_reflect.dropna(axis='index', inplace=True)

			for t_source in t_range:
				ref_data.at[t_source, 'Emissivity'] = integrate.trapz(mean_reflect[str(t_source) +'_measured'], x=mean_reflect.index)/integrate.trapz(mean_reflect[str(t_source) +'_spectrum'], x=mean_reflect.index)
				ref_data.loc[t_source, 'Std. Dev.'] = integrate.trapz(mean_reflect[str(t_source) +'_measured_std'], x=mean_reflect.index)/integrate.trapz(mean_reflect[str(t_source) +'_spectrum'], x=mean_reflect.index)	

			# ref_data = pd.DataFrame(ref_data).reset_index()
			ref_data = ref_data.astype('float64')
			ref_data['Emissivity'] = ref_data['Emissivity'].round(3)
			ref_data['Std. Dev.'] = ref_data['Std. Dev.'].round(3)

			ref_data.index.rename('Source Temperature [K]', inplace=True)
			ref_data = ref_data.reset_index()
			ref_data.to_html(f'{data_dir}{material}/FTIR/IS/{material}_Emissivity.html', index=False,border=0)
			
			fig = go.Figure()
			ref_data.set_index('Source Temperature [K]', inplace=True)
			plot_mean_data(ref_data)

			plot_dir = f'../03_Charts/{material}/FTIR/IS/'

			if not os.path.exists(plot_dir):
				os.makedirs(plot_dir)

			# print('Plotting Chart')
			format_and_save_plot(f'{plot_dir}{material}_Emissivity.html',material)
			mat_status_df.loc[material, 'Emissivity'] = True
			print(material, mat_status_df.loc[material, 'Emissivity'])

mat_status_df.to_csv('Utilities/material_status.csv', index_label = 'material')
print()
