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

# Define other general plot parameters
label_size = 20
tick_size = 18
line_width = 2
legend_font = 14
fig_width = 10
fig_height = 8

h = 6.62607015e-34 # J-s (Planck's Constant)
c = 299702547 # m/s (speed of light in air)
kB = 1.380649e-23 # J/K (Boltzmann constant)
sig = 5.67e-8 # W/m2K4

data_dir = '../01_Data/'
save_dir = '../03_Charts/'

def create_1plot_fig():
	# Define figure for the plot
	fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
	#plt.subplots_adjust(left=0.08, bottom=0.3, right=0.92, top=0.95)

	# Reset values for x & y limits
	x_min, x_max, y_min, y_max = 0, 0, 0, 0

	return(fig, ax1, x_min, x_max, y_min, y_max)

def plot_data(df):

	print(df)
	temp_dict = {'600':'k', '800': 'b', '1000': 'r', '1200': 'g', '1400': '0.25', '1600': 'c', '1800': 'y', '2000' :'0.5'}

	thickness = [25.4*float(i)/1000 for i in df.columns]
	log_mean = list(df.loc['lMean', :])


	print(thickness)
	print(log_mean)
	# df_spectrum = df.filter(axis = 'columns', regex = 'spectrum_norm')

	# print(df_spectrum)

	# for i in df_spectrum.columns:
	# 	temp = i.split('.0_spectrum_norm')[0]
	# 	ax1.plot(df_spectrum.index, df_spectrum[i], color=temp_dict[temp], ls='-', marker=None, label = f'{temp} K')
	# ax1.fill_between(df.index, df_low, df_high, color=hf_dict[hf], alpha=0.2)

	# df_emissivity = df['measured'] 
	# ax1.plot(df_emissivity.index, df_emissivity, color='k', ls='-')
	# df_std = (2*np.sqrt((df['measured_std']/df['measured'])**2+(0.04**2)))*df['measured']
	# df_low = df['measured'] - df_std
	# df_high = df['measured'] + df_std
	# ax1.fill_between(df_emissivity.index, df_low, df_high, color='k', alpha=0.2)

	ax1.plot(thickness, log_mean, marker = 'o', color='k', markersize = 7, ls = 'None')

	my, by, ry, smy, sby, s2, rmb, smb = lsqfity(thickness, log_mean)

	line_low = by
	line_high = my*0.004+by
	ax1.plot([0.0000, 0.004], [line_low, line_high], color='k', ls = '-')

	# y_max = max(df)
	# y_min = min(df)

	# x_max = max(df.index)
	# x_min = min(df.index)
	
	# return(y_min, y_max, x_min, x_max)
	return ry

def plot_mean_data(df):
	# fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:,mean_col],error_y=dict(type='data', array=2*df.iloc[:,std_col],),mode='markers',name=data_lab, marker=dict(color=c, size=8)))
	fig.add_trace(go.Scatter(x=df.index, y=df['Emissivity'],error_y=dict(type='data', array=2*df['Std. Dev.']),mode='markers',name='Mean', marker=dict(color='blue', size=8)))

	return()

def reindex_data(data, min_x, max_x, dx):
	data = data.reindex(data.index.union(np.arange(min_x, max_x+dx, dx)))
	data = data.astype('float64')
	data.index = data.index.astype('float64')
	data = data.interpolate(method='cubic')

	data = data.loc[np.arange(min_x, max_x+dx, dx)]

	return data

def format_and_save_plot(file_loc, material):
	# fig.update_layout(yaxis_title='Total Hemispherical Emissivity', title ='Specific Heat')

	ax1.set_ylim(bottom=0, top=2) # 1.4e11
	ax1.set_xlim(left=0, right=0.004)
	ax1.set_xlabel('Thickness (m)', fontsize=label_size)
	ax1.set_ylabel('-ln($T$)', fontsize=label_size)

	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)

	ax1.set_position([0.15, 0.18, 0.77, 0.72])

	# ax1.annotate('-ln(T) = ')

	# ax1.autoscale(enable = True, axis = 'both')

	#Get github hash to display on graph
	repo = git.Repo(search_parent_directories=True)
	sha = repo.head.commit.hexsha
	short_sha = repo.git.rev_parse(sha, short=True)

	# fig.add_annotation(dict(font=dict(color='black',size=15),
	# 									x=1,
	# 									y=1.02,
	# 									showarrow=False,
	# 									text="Repository Version: " + short_sha,
	# 									textangle=0,
	# 									xanchor='right',
	# 									xref="paper",
	# 									yref="paper"))

	# Add legend
	handles1, labels1 = ax1.get_legend_handles_labels()
	# plt.legend(handles1, labels1, loc = 'best', fontsize=16, handlelength=2, frameon=True, framealpha=1.0, ncol=1)

	# Save plot to file
	# fig.update_layout(xaxis_title='Source Temperature [K]', font=dict(size=18))
	# fig.write_html(file_loc,include_plotlyjs="cdn")
	plt.savefig(file_loc)
	plt.close()

	print()

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
	# if material != 'Luan_Panel':
	# 	continue
	if material != 'Nylon':
		continue	
	trans_sample_df = pd.DataFrame()
	trans_trap_df = pd.DataFrame()
	sample_df = pd.DataFrame()
	trap_df = pd.DataFrame()
	if os.path.isdir(f'{data_dir}{material}/FTIR/IS'):
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
			wl = [round(10000000/i, 1) for i in mean_trans.index]
			mean_trans['wavelength'] = wl # wavelength in nm
			mean_trans.set_index('wavelength', inplace=True)

			mean_trans['wl_m'] = mean_trans.index*1e-9 # convert from nm to m
			trans_data = pd.DataFrame(index = t_range, columns = th_list)
			for t_source in t_range:
				mean_trans[f'{t_source}_spectrum'] = (2*h*c**2)/(mean_trans['wl_m']**5*(np.exp((h*c)/(mean_trans['wl_m']*kB*t_source))-1)) # Planck's Law [W/m^3]
				for y in th_list:
					mean_trans[f'{t_source}_measured_{y}'] = (2*h*c**2*mean_trans[y])/(mean_trans['wl_m']**5*(np.exp((h*c)/(mean_trans['wl_m']*kB*t_source))-1)) # Weighted spectral transmission
					# mean_trans[str(t_source) +'_measured_std'] = (2*h*c**2*mean_trans[f'{x}_std'])/(mean_trans['wl_m']**5*(math.exp((h*c)/(mean_trans['wl_m']*kB*t_source))-1))
					trans_data.at[t_source, y] = integrate.trapezoid(mean_trans[f'{t_source}_measured_{y}'], x=mean_trans.index)/integrate.trapezoid(mean_trans[f'{t_source}_spectrum'], x=mean_trans.index) # Mean total transmittance w.r.t. thickness Eq 6 in Linteris [10.1002/fam.1113]
					# ref_data.loc[t_source, 'Std. Dev.'] = integrate.trapezoid(mean_reflect[str(t_source) +'_measured_std'], x=mean_reflect.index)/integrate.trapezoid(mean_reflect[str(t_source) +'_spectrum'], x=mean_reflect.index)
		trans_data.loc['Mean', :] = trans_data.mean(axis='index')
		for i in trans_data.columns:
			trans_data.at['lMean', i] = -np.log(trans_data.loc['Mean', i]) # -log of mean transmittance
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
			wl = [round(10000000/i, 1) for i in mean_reflect.index]
			mean_reflect['wavelength'] = wl # wavelength in nm
			mean_reflect.set_index('wavelength', inplace=True)

			min_x = math.ceil(min(mean_reflect.index))
			max_x = math.floor(max(mean_reflect.index))
			mean_reflect = reindex_data(mean_reflect, min_x, max_x, 0.1)

			if fid_list_trans:
				d = max(list(mean_trans.columns))
				mean_trans_d = pd.DataFrame(index = wl)
				mean_trans_d['transmittance'] = mean_trans[d]
			else:
				mean_trans_d = pd.DataFrame(index = wl)
				mean_trans_d['transmittance'] = [0]*len(wl)
			mean_trans_d = reindex_data(mean_trans_d, min_x, max_x, 0.1)

			mean_reflect['measured'] = 1-((mean_reflect['Sample']-mean_reflect['LT'])/(1-mean_reflect['LT'])) - mean_trans_d['transmittance']
			# uncertainty propagation for measured reflectance
			mean_reflect['measured_std_comp_1'] = np.sqrt(mean_reflect['Sample_std']**2+mean_reflect['LT_std']**2)
			mean_reflect['measured_std_comp_1_rel'] = mean_reflect['measured_std_comp_1']/(mean_reflect['Sample']-mean_reflect['LT'])
			mean_reflect['measured_std_comp_2_rel'] = mean_reflect['LT_std']/mean_reflect['LT']
			mean_reflect['measured_std'] = ((mean_reflect['Sample']-mean_reflect['LT'])/(1-mean_reflect['LT']))*np.sqrt(mean_reflect['measured_std_comp_1_rel']**2 + mean_reflect['measured_std_comp_2_rel']**2)

			# for ind in mean_reflect.index:
			# 	i = ind/1000000000 # convert from nm to m
			# 	for t_source in t_range:
			# 		mean_reflect.loc[ind, str(t_source) +'_spectrum'] = (2*h*c**2)/(i**5*(math.exp((h*c)/(i*kB*t_source))-1)) # Planck's Law [W/m^3]
			# 		mean_reflect.loc[ind, str(t_source) +'_measured'] = (2*h*c**2*mean_reflect.loc[ind, 'measured'])/(i**5*(math.exp((h*c)/(i*kB*t_source))-1))
			# 		mean_reflect.loc[ind, str(t_source) +'_measured_std'] = (2*h*c**2*mean_reflect.loc[ind, 'measured_std'])/(i**5*(math.exp((h*c)/(i*kB*t_source))-1))

			ref_data = pd.DataFrame(index = t_range, columns = ['Emissivity', 'Std. Dev.'])
			mean_reflect.sort_index(inplace=True)

			# for t_source in t_range:
			# 	mean_reflect[str(t_source) +'_spectrum_norm'] = mean_reflect[str(t_source) +'_spectrum']/integrate.trapezoid(mean_reflect[str(t_source) +'_spectrum'], x=mean_reflect.index) # Planck's Law [W/m^3]
			# 	ref_data.at[t_source, 'Emissivity'] = integrate.trapezoid(mean_reflect[str(t_source) +'_measured'], x=mean_reflect.index)/integrate.trapezoid(mean_reflect[str(t_source) +'_spectrum'], x=mean_reflect.index)
			# 	ref_data.loc[t_source, 'Std. Dev.'] = integrate.trapezoid(mean_reflect[str(t_source) +'_measured_std'], x=mean_reflect.index)/integrate.trapezoid(mean_reflect[str(t_source) +'_spectrum'], x=mean_reflect.index)

			# ref_data = pd.DataFrame(ref_data).reset_index()
			ref_data = ref_data.astype('float64')
			ref_data['Emissivity'] = ref_data['Emissivity'].round(3)
			# ref_data['Std. Dev.'] = ref_data['Std. Dev.'].round(3)

			# ref_data.index.rename('Source Temperature [K]', inplace=True)
			# ref_data.to_html(f'{data_dir}{material}/FTIR/IS/{material}_Emissivity.html', classes='col-xs-12 col-sm-6')

			fig, ax1, x_min, x_max, y_min, y_max = create_1plot_fig()
			# ymin, ymax, xmin, xmax = plot_data(trans_data)
			ry = plot_data(trans_data)
			print(ry)

			# y_min = max(ymin, y_min)
			# x_min = max(xmin, x_min)
			# y_max = max(ymax, y_max)
			# x_max = max(xmax, x_max)

			# inc = y_inc_dict[n]

			# ylims[0] = 0 #y_min - abs(y_min * 0.1)
			# ylims[1] = y_max * 1.1
			# xlims[0] = x_min
			# xlims[1] = x_max

			plot_dir = f'../03_Charts/{material}/FTIR/IS/'

			if not os.path.exists(plot_dir):
				os.makedirs(plot_dir)

			print('Plotting Chart')
			format_and_save_plot(f'{plot_dir}{material}_Absorption_coefficient.png',material)