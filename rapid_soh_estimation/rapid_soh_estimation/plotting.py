
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from common_methods import *

from rapid_soh_estimation.rapid_soh_estimation.slowpulse import get_pulse_model_dataset
from rapid_soh_estimation.rapid_soh_estimation.cc_segment import get_cc_model_dataset
from rapid_soh_estimation.rapid_soh_estimation.quantization import TFL_Model_Wrapper
from rapid_soh_estimation.rapid_soh_estimation.postprocessing import extract_cccv_charge


def load_all_data_for_plotting():
    data = {
        'LFP':{
            'pulse':{'chg':None, 'dchg':None},
            'cc':None
        },
        'NMC':{
            'pulse':{'chg':None, 'dchg':None},
            'cc':None
        }
    }
    for chemistry in ['LFP', 'NMC']:
        for pulse_type in ['chg', 'dchg']:
            data[chemistry]['pulse'][pulse_type] = get_pulse_model_dataset(chemistry=chemistry, pulse_type=pulse_type)
        data[chemistry]['cc'] = get_cc_model_dataset(chemistry=chemistry)

    return data

def load_saved_models():
    '''returns dic containing the latest saved model for each each test type'''

    # dictionary to return containing each model type and its quantized version
    ret_dic = {
        'LFP': {
            'pulse_dchg': {'full':None, 'tfl':None, 'input_scaler':None, 'output_scaler':None},
            'pulse_chg': {'full':None, 'tfl':None, 'input_scaler':None, 'output_scaler':None},
            'cc': {'full':None, 'tfl':None, 'input_scaler':None, 'output_scaler':None},
        },
        'NMC': {
            'pulse_dchg': {'full':None, 'tfl':None, 'input_scaler':None, 'output_scaler':None},
            'pulse_chg': {'full':None, 'tfl':None, 'input_scaler':None, 'output_scaler':None},
            'cc': {'full':None, 'tfl':None, 'input_scaler':None, 'output_scaler':None},
        },
    }

    for chemistry in ['LFP', 'NMC']:
        #region: go through slowpulse models
        for pulse_type in ['chg', 'dchg']:
            dir_temp = dir_results.joinpath("saved_models", chemistry, 'slowpulse', pulse_type)
            folder_latest_models = get_latest_models_folder(dir_temp, look_for='*.h5')

            files_all_models = sorted ( folder_latest_models.glob('*.h5') )
            if len(files_all_models) == 0:
                raise ValueError(f"There are no saved models at: {folder_latest_models}")
            file_newest_model = files_all_models[-1]
            ret_dic[chemistry][f'pulse_{pulse_type}']['full'] = tf.keras.models.load_model(file_newest_model)
            try:
                file_newest_tfl_model = sorted(folder_latest_models.glob('*.tflite'))[-1]
                ret_dic[chemistry][f'pulse_{pulse_type}']['tfl'] = TFL_Model_Wrapper(file_newest_tfl_model)
            except:
                ret_dic[chemistry][f'pulse_{pulse_type}']['tfl'] = None
            file_input_scaler = sorted(folder_latest_models.glob("Model_Scaler_Input*"))[-1]
            ret_dic[chemistry][f'pulse_{pulse_type}']['input_scaler'] = pickle.load(open(file_input_scaler, 'rb'))
            file_output_scaler = sorted(folder_latest_models.glob("Model_Scaler_Output*"))[-1]
            ret_dic[chemistry][f'pulse_{pulse_type}']['output_scaler'] = pickle.load(open(file_output_scaler, 'rb'))
        #endregion

        #region: go through cc models
        dir_temp = dir_results.joinpath("saved_models", chemistry, 'cc')
        folder_latest_models = get_latest_models_folder(dir_temp, look_for='*.h5')
        
        files_all_models = sorted ( folder_latest_models.glob('*.h5') )
        if len(files_all_models) == 0:
            raise ValueError(f"There are no saved models at: {folder_latest_models}")
        file_newest_model = files_all_models[-1]
        ret_dic[chemistry]['cc']['full'] = tf.keras.models.load_model(file_newest_model)
        try:
            file_newest_tfl_model = sorted(folder_latest_models.glob('*.tflite'))[-1]
            ret_dic[chemistry]['cc']['tfl'] = TFL_Model_Wrapper(file_newest_tfl_model)
        except:
            ret_dic[chemistry]['cc']['tfl'] = None
        file_input_scaler = sorted(folder_latest_models.glob("Model_Scaler_Input*"))[-1]
        ret_dic[chemistry]['cc']['input_scaler'] = pickle.load(open(file_input_scaler, 'rb'))
        file_output_scaler = sorted(folder_latest_models.glob("Model_Scaler_Output*"))[-1]
        ret_dic[chemistry]['cc']['output_scaler'] = pickle.load(open(file_output_scaler, 'rb'))
        #endregion

    return ret_dic



def plot_rpt_section_pulses(pulse_type:str='chg', stack_horizontal:bool=True, save:bool=False):
	cell_id = 2

	file_rpt = dir_preprocessed_data.joinpath("rpt_data", f'rpt_cell_{cell_id:02d}_part0.pkl')
	df_rpt = pickle.load(open(file_rpt, 'rb'))

	#region: create fig, axes
	fig, axes = None, None
	if stack_horizontal:
		fig = plt.figure(figsize=(8,3), constrained_layout=True)
		gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[2,3], wspace=0.05) 
		gs_inner = gs_outer[1].subgridspec(3, 2, width_ratios=[19,1])
		
		axes = [
			fig.add_subplot(gs_outer[:,0]),     # Portion of RPT voltage profile
			fig.add_subplot(gs_inner[0, 0]), 	# Q fade for pulse at 20% SOC
			fig.add_subplot(gs_inner[1, 0]), 	# Q fade for pulse at 50% SOC
			fig.add_subplot(gs_inner[2, 0]),    # Q fade for pulse at 90% SOC
			fig.add_subplot(gs_inner[:,1]),     # colorbar
		]
	else:
		fig = plt.figure(figsize=(8,4), constrained_layout=True)
		gs_outer = GridSpec(2, 1, figure=fig, height_ratios=[2,3], hspace=0.05) 
		gs_inner = gs_outer[1].subgridspec(1, 4, width_ratios=[20,20,20,1], wspace=0.05)
		
		axes = [
			fig.add_subplot(gs_outer[0,:]),     # Portion of RPT voltage profile
			fig.add_subplot(gs_inner[0]), 		# Q fade for pulse at 20% SOC 
			fig.add_subplot(gs_inner[1]), 		# Q fade for pulse at 50% SOC 
			fig.add_subplot(gs_inner[2]),   	# Q fade for pulse at 90% SOC 
			fig.add_subplot(gs_inner[3]),     	# colorbar
		]
	#endregion

	#region: plot RPT profile
	# plot continuous RPT voltage
	rpt_time = df_rpt.loc[(df_rpt['RPT Number'] == 0), 'Time (s)'].values
	rpt_voltage = df_rpt.loc[(df_rpt['RPT Number'] == 0), 'Voltage (V)'].values
	filt_idxs = np.where((rpt_voltage >= 3.15) & (rpt_voltage <= 3.5) )[0]
	split_idxs = [i for i in range(len(filt_idxs)) if not filt_idxs[i] == filt_idxs[i-1]+1 ]
	filt_idxs = filt_idxs[split_idxs[0]:split_idxs[1]]

	axes[0].plot(rpt_time[filt_idxs]/60, rpt_voltage[filt_idxs], '-', color='grey', linewidth=2.5)
	for soc in [20,50,90]:
		df_filt = df_rpt.loc[
			(df_rpt['RPT Number'] == 0) & \
			(df_rpt['Segment Key'] == 'slowpulse') & \
			(df_rpt['Pulse Type'] == pulse_type) & \
			(df_rpt['Pulse SOC'] == soc)]
		axes[0].plot(df_filt['Time (s)'].values/60, df_filt['Voltage (V)'].values, '-', color='C0', linewidth=2.5)
	axes[0].set_xlabel("Time [min]")
	axes[0].set_ylabel("Voltage [V]")
	#endregion

	#region: plot capacity fade for pulses at each SOC
	data = pickle.load(open(dir_processed_data.joinpath("LFP", 'slowpulse', 'processed_data_slowpulse_0.pkl'), 'rb'))
	data['voltage_rel'] = np.asarray([data['voltage'][i] - data['voltage'][i][0] for i in range(len(data['voltage']))])

	scm = mpl.cm.ScalarMappable(cmap=mpl.cm.Blues, norm=mpl.colors.Normalize(vmin=np.min(data['q_dchg'])*0.9, vmax=np.max(data['q_dchg'])))
	scm.set_array([])
	for i, soc in enumerate([20,50,90]):
		filt_idxs = np.where( (data['cell_id'] == cell_id) & (data['soc'] == soc) & (data['pulse_type'] == pulse_type))[0]
		for j in range(len(filt_idxs)):
			axes[i+1].plot(data['voltage_rel'][filt_idxs[j]]*1000, 'o', color=scm.to_rgba(data['q_dchg'][filt_idxs[j]]), alpha=0.8, markersize=2)
		if pulse_type == 'chg':
			axes[i+1].set_ylim([-5,50])
		elif pulse_type == 'dchg':
			axes[i+1].set_ylim([-50,5])

	# Add colorbar
	fig.colorbar(scm, cax=axes[4], label='Capacity [Ah]')
	#endregion

	#region: set axis labels
	if stack_horizontal:
		
		for i, soc in enumerate(['20%', '50%', '90%']):
			if pulse_type == 'chg':
				axes[i+1].annotate("{} SOC".format(soc), xy=(0.77,0.7), xycoords='axes fraction', fontsize=10 )
			else:
				axes[i+1].annotate("{} SOC".format(soc), xy=(0.76,0.1), xycoords='axes fraction', fontsize=10 )
		axes[2].set_ylabel("Relative Voltage [mV]")
		axes[3].set_xlabel("Time [s]")
	else:
		for i, soc in enumerate(['20%', '50%', '90%']):
			if pulse_type == 'chg':
				axes[i+1].annotate("{} SOC".format(soc), xy=(0.63,0.9), xycoords='axes fraction', fontsize=10 )
			else:
				axes[i+1].annotate("{} SOC".format(soc), xy=(0.63,0.03), xycoords='axes fraction', fontsize=10 )
		axes[1].set_ylabel("Relative Voltage [mV]")
		axes[1].set_xlabel("Time [s]")
		axes[2].set_xlabel("Time [s]")
		axes[3].set_xlabel("Time [s]")
	#endregion

	if save: 
		dir_save = dir_figures.joinpath("raw", "Input Features")
		dir_save.mkdir(parents=True, exist_ok=True)
		filename = f"RPT Overview - {pulse_type.upper()} Pulse ({'Horizontal' if stack_horizontal else 'Vertical'}).png"
		fig.savefig(dir_save.joinpath(filename), dpi=300)
		print(f"Figure saved to: {dir_save.joinpath(filename)}")
	plt.show()

def plot_pulse_capacity_fade(pulse_type:str='chg', save:bool=False):
	cell_id_to_plot = 31

	data = pickle.load(open(dir_processed_data.joinpath("LFP", 'slowpulse', 'processed_data_slowpulse_0.pkl'), 'rb'))
	data['voltage_rel'] = np.asarray([data['voltage'][i] - data['voltage'][i][0] for i in range(len(data['voltage']))])
	qs = data['q_dchg'][np.where((data['cell_id'] == cell_id_to_plot) & (data['pulse_type'] == pulse_type))]
	sohs = qs / np.max(qs) * 100 
	vrels = data['voltage_rel'][np.where((data['cell_id'] == cell_id_to_plot) & (data['pulse_type'] == pulse_type))]
	t = np.arange(0,100,1)

	scm = mpl.cm.ScalarMappable(cmap=mpl.cm.Blues, norm=mpl.colors.Normalize(vmin=np.min(sohs)*0.9, vmax=np.max(sohs)))
	fig, ax = plt.subplots(figsize=(4,3))
	vrels = vrels[np.argsort(sohs)]
	for i in range(len(vrels)):
		ax.plot(vrels[i]*1000, 'o', color=scm.to_rgba(sohs[i]), alpha=0.8, markersize=2)
	if pulse_type == 'chg':
		ax.set_ylim([-5,50])

	fig.tight_layout(pad=0.8)
	fig.subplots_adjust(right=0.80, bottom=0.15, left=0.14, top=0.95)
	cbar_ax = fig.add_axes([0.82, 0.22, 0.03, 0.6])
	fig.colorbar(scm, cax=cbar_ax)
	cbar_ax.set_title('SOH [%]', loc='left', pad=12, fontsize=10)

	ax.set_xlabel("Time [s]", fontsize=10)
	ax.set_ylabel("Relative Voltage [mV]", fontsize=10)
	if save: 
		dir_save = dir_figures.joinpath("raw", "Input Features")
		dir_save.mkdir(parents=True, exist_ok=True)
		filename = f"Capacity Fade - {pulse_type.upper()} Pulse.png"
		fig.savefig(dir_save.joinpath(filename), dpi=300)
		print(f"Figure saved to: {dir_save.joinpath(filename)}")
	plt.show()

def plot_cc_input_comparison(save:bool=False):
	"""Plots the CC voltage segments sampled at different SOCs along the 1C CC-CV charge profile

	Args:
		save (bool, optional): Whether to save the figure. If True, the path location of the saved plot is printed. Defaults to False
	"""
		
	cm = mpl.cm.Blues
	v_bounds = (3, 3.5)
	seg_idxs_to_plot = [0,2,4]
	cell_id_to_plot = 2
	rpt_num_to_plot = 0

	#region: get all data
	# get rpt data for this cell
	rpt_data = load_preprocessed_data(get_preprocessed_data_files(dir_preprocessed_data, data_type='rpt', cell_id=cell_id_to_plot))
	# filter to specified RPT number 
	rpt_data = rpt_data.loc[(rpt_data['RPT Number'] == rpt_num_to_plot)]
	# get cc-cv charge data from rpt data
	df_cc = extract_cccv_charge(rpt_data=rpt_data, plot_interpolation=False)

	# get data for sampled voltage segments and corresponding time
	data = load_processed_data(data_type='cc')
	filt_idxs = np.where((data['cell_id'] == cell_id_to_plot) & (data['rpt'] == rpt_num_to_plot))

	segment_vs = data['voltage'][filt_idxs]		# each row = a sample of 600 seconds
	segment_ts = np.zeros_like(segment_vs)
	for i in range(len(segment_vs)):
		t_start = df_cc.loc[df_cc['Voltage (V)'] == segment_vs[i,0], 'Time (s)'].values[0]
		segment_ts[i] = np.arange(t_start, t_start+len(segment_vs[0]), 1)
	#endregion

	#region: plot CCCV and subsamples
	fig, axes = plt.subplots(figsize=(4,4), nrows=len(seg_idxs_to_plot), ncols=1, sharey=True, sharex=True)    
	for i, seg_idx in enumerate(seg_idxs_to_plot):
		df_filt = df_cc.loc[(df_cc['Voltage (V)'] >= v_bounds[0]) & (df_cc['Voltage (V)'] <= v_bounds[1])]
		soc = df_v_vs_soc_1c_chg.loc[df_v_vs_soc_1c_chg['v'] <= segment_vs[seg_idx, 0], 'soc'].values[-1]

		# plot CCCV voltage profile
		axes[i].plot(
			df_filt['Time (s)'].values / 60, 
			df_filt['Voltage (V)'].values, 
			'-', linewidth=2, color='grey')
			
		# plot voltage subsample
		axes[i].plot(
			segment_ts[seg_idx] / 60, 
			segment_vs[seg_idx], 
			'-', linewidth=3, color=cm(0.8), 
			label='{} minutes of data starting @ ~{}% SOC'.format(int(len(segment_ts[seg_idx])/60), int(soc)))
		
		axes[i].fill_between(
			segment_ts[seg_idx] / 60, 
			v_bounds[0], v_bounds[1], 
			color=cm(0.2))
		axes[i].set_ylim(v_bounds)
		axes[i].legend(fontsize=8, loc='lower center')	
	axes[1].set_ylabel("Voltage [V]")
	axes[2].set_xlabel("Time [min]")
	fig.tight_layout(pad=1.0)
	#endregion

	if save: 
		dir_save = dir_figures.joinpath("raw", "Input Features")
		dir_save.mkdir(parents=True, exist_ok=True)
		filename = f"CC Voltage Segment Overview.png"
		plt.savefig(dir_save.joinpath(filename), dpi=300)
		print(f"Figure saved to: {dir_save.joinpath(filename)}")
	plt.show()



def plot_feature_importance(dir_hyperparam_results:Path, dir_save:Path=None):
	"""
	Plots the feature importance of the pulse signal using linear models and a random forest regressor. 
	See the 'feature_importance.ipynb' notebook for an implementation example.

	Args:
		dir_hyperparam_results (Path): Path object to folder containing hyperparameter results
		dir_save (Path, optional): If provided, the plot will be saved in this folder. Defaults to None.
	"""

	plot_params = {
		'ridge': {'legend_key':'Ridge', 'color':'C0',},
		'lasso': {'legend_key':'Lasso', 'color':'C1',},
		'elasticnet': {'legend_key':'ElasticNet', 'color':'C2',},
		'randomforest': {'legend_key':'RandomForest', 'color':'C4',},
	}

	#region: prepare data for plotting
	#region: create modeling dataset (charge pulse, all SOCs)
	# load pulse and cc data
	cc_data = load_processed_data(data_type='cc')
	pulse_data = load_processed_data(data_type='slowpulse')
	# create modeling data
	all_data = deepcopy(pulse_data)
	idxs = np.where((all_data['pulse_type'] == 'chg'))
	for k in all_data.keys():
		all_data[k] = all_data[k][idxs]
	all_data['voltage_rel'] = np.asarray([v - v[0] for v in all_data['voltage']])
	modeling_data = create_modeling_data(all_data=all_data, input_feature_keys=['voltage_rel'])
	X = modeling_data['model_input']
	y = modeling_data['model_output'][:,0].reshape(-1,1)
	#endregion

	#region: get CV splits
	cv_splitter = Custom_CVSplitter(n_splits=3, split_type='group_id', rand_seed=random_state)
	cv_splits = list(cv_splitter.split(
		X = modeling_data['model_input'], 
		y = modeling_data['model_output'][:,0], 
		cell_ids = modeling_data['cell_id']))
	#endregion

	# use a fixed split for plotting
	train_idxs, test_idxs = cv_splits[0]

	# standardize data using only training samples
	scaler_X = StandardScaler().fit(X[train_idxs])
	scaler_y = StandardScaler().fit(y[train_idxs])
	X_sc = scaler_X.transform(X)
	y_sc = scaler_y.transform(y)
	#endregion

	fig = plt.figure(figsize=(6.25,3.75))
	gs = GridSpec(nrows=2, ncols=2, height_ratios=[3,2], width_ratios=[1,1])
	axes = [
		fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
		fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
	]
	# axes[2].sharex(axes[0])
	# axes[3].sharex(axes[1])
	axes[0].set_xticklabels([])
	axes[1].set_xticklabels([])

	for model_type in plot_params.keys():
		file_study = list(dir_hyperparam_results.glob(f'*{model_type}_onlySOH.pkl*'))[0]
		study = pickle.load(open(file_study, 'rb'))
		# create optimal model from saved parameters values
		model = None
		if model_type == 'ridge': model = Ridge(**study.best_trial.params)
		if model_type == 'lasso': model = Lasso(**study.best_trial.params)
		if model_type == 'elasticnet': model = ElasticNet(**study.best_trial.params)
		if model_type == 'randomforest': model = RandomForestRegressor(**study.best_trial.params)
		# fit model to training data
		model.fit(X_sc[train_idxs], y_sc[train_idxs])

		x_vals = None
		if model_type == 'randomforest':
			x_vals = model.feature_importances_
			axes[1].plot(
				x_vals*100, '-', linewidth=2,
				color=plot_params[model_type]['color'],
				label=plot_params[model_type]['legend_key'])
		else:
			x_vals = model.coef_
			if len(x_vals.shape) == 2: x_vals = np.ravel(x_vals)
			axes[0].plot(
				x_vals, '-', linewidth=2,
				color=plot_params[model_type]['color'],
				label=plot_params[model_type]['legend_key'])

		if model_type == 'ridge':
			# region: overlay feature importance on pulse profile
			pulse_v = X[train_idxs][0,:]*1000
			axes[2].plot(pulse_v, 'k', linewidth=2)
			ybounds = [np.min(pulse_v), np.max(pulse_v)]
			ybounds[0] -= (ybounds[1]-ybounds[0])*0.05
			ybounds[1] += (ybounds[1]-ybounds[0])*0.05

			norm_coeff = abs(x_vals)
			norm_coeff = (norm_coeff - np.min(norm_coeff)) / (np.max(norm_coeff) - np.min(norm_coeff))
			for i in np.arange(0,100,1):
				axes[2].fill_betweenx(
					ybounds, i-0.5, i+0.5, 
					color=plot_params[model_type]['color'], alpha=norm_coeff[i])
			axes[2].set_xlim([-5,104])
			axes[2].set_ylim(ybounds)
			#endregion
		elif model_type == 'randomforest':
			# region: overlay feature importance on pulse profile
			pulse_v = X[train_idxs][0,:]*1000
			axes[3].plot(pulse_v, 'k', linewidth=2)
			ybounds = [np.min(pulse_v), np.max(pulse_v)]
			ybounds[0] -= (ybounds[1]-ybounds[0])*0.05
			ybounds[1] += (ybounds[1]-ybounds[0])*0.05

			norm_coeff = abs(x_vals)
			norm_coeff = (norm_coeff - np.min(norm_coeff)) / (np.max(norm_coeff) - np.min(norm_coeff))
			for i in np.arange(0,100,1):
				axes[3].fill_betweenx(
					ybounds, i-0.5, i+0.5, 
					color=plot_params[model_type]['color'], alpha=norm_coeff[i])
			axes[3].set_xlim([-5,104])
			axes[3].set_ylim(ybounds)
			#endregion
	
	#region: set labels
	axes[0].plot([0,99],[0,0], '--', color='k')
	axes[0].set_ylabel("Model Coefficients [-]")
	axes[0].set_title('Linear Models')
	axes[0].legend(fontsize=8, loc='lower right')

	axes[1].set_ylabel("Feature Importance [%]")
	axes[1].set_title('Random Forest')

	axes[2].set_xlabel("Time [s]")
	axes[2].set_ylabel("Rel. Voltage [mV]")
	axes[3].set_xlabel("Time [s]")
	axes[3].set_ylabel("Rel. Voltage [mV]")
	fig.align_ylabels()
	#endregion
	
	fig.tight_layout(h_pad=0.5, w_pad=1.2)
	if dir_save is not None:
		filename = dir_save.joinpath("FeatureImportance_Pulse.pdf")
		fig.savefig(filename, dpi=300)
	plt.show()

def plot_feature_importance_model_performance(dir_hyperparam_results:Path, dir_save:Path=None):
	"""
	Plots the baseline model perforamnce for linear models, random forest regressor, and MLP
	See the 'feature_importance.ipynb' notebook for an implementation example.

	Args:
		dir_hyperparam_results (Path): Path object to folder containing hyperparameter results
		dir_save (Path, optional): If provided, the plot will be saved in this folder. Defaults to None.
	"""

	def _evaluate_model(model, X, y, cv_splits) -> dict:
		"""Evaluates the train and test accuracy of a given model

		Args:
			model (-): A model to be tested. Must have .fit() and .predict() functions
			X (_type_): Training data
			y (_type_): Test data
			cv_splits (_type_): A list of indices for train/test splits at each CV fold

		Returns:
			dict: Returns the average and std of the train and test error. Follows the following structure: {'train':(avg, std), 'test':(avg, std)}
		"""

		# average the loss over all cross-validation splits
		test_err = []
		train_err = []
		for train_idxs, test_idxs in cv_splits:
			# standardize input and output data (using only the training data to create the scaler)
			scaler_X = StandardScaler().fit(X[train_idxs])
			scaler_y = StandardScaler().fit(y[train_idxs])
			X_sc = scaler_X.transform(X)
			y_sc = scaler_y.transform(y)
			# fit model to scaled input and output data
			model.fit(X_sc[train_idxs], y_sc[train_idxs])
			
			# get test and train predictions
			yhat_train = model.predict(X_sc[train_idxs])
			yhat_test = model.predict(X_sc[test_idxs])
			train_err.append( mean_absolute_percentage_error(y_sc[train_idxs], yhat_train))
			test_err.append( mean_absolute_percentage_error(y_sc[test_idxs], yhat_test))

		# return average and std of mape
		return {
			'test':(np.average(np.asarray(test_err)), np.std(np.asarray(test_err))),
			'train':(np.average(np.asarray(train_err)), np.std(np.asarray(train_err)))
		}
	
	def _create_mlp_model(n_hlayers:int, n_neurons:int, act_fnc:str, opt_fnc:str, learning_rate:float, input_shape=(100,), output_shape=(7,)) -> keras.models.Sequential:
		"""Builds a Keras neural network model (MLP) using the specified parameters. The model is optimized for accuracy. Make sure model outputs (if multiple target) are normalized, otherwise optimization will be biased towards one target variable.

		Args:
			n_hlayers (int): Number of fully-connected hidden layers
			n_neurons (int): Number of neurons per hidden layer
			act_fnc (str): Activation function to use (\'tanh\', \'relu\', etc)
			opt_fnc (str): {\'sgd\', \'adam\'} Optimizer function to use 
			learning_rate (float): Learning rate
			input_shape (int, optional): Input shape of model. Defaults to (100,).
			output_shape (int, optional): Output shape of model. Default to (7,).
		Raises:
			ValueError: _description_

		Returns:
			keras.models.Sequential: compiled Keras model
		"""

		# add input layer to Sequential model
		model = keras.models.Sequential()
		model.add( keras.Input(shape=input_shape) )

		# add hidden layers
		for i in range(n_hlayers):
			model.add( keras.layers.Dense(units=n_neurons, activation=act_fnc) )
			
		# add output layer
		model.add( keras.layers.Dense(output_shape[0]) )

		# compile model with chosen metrics
		opt = None
		if opt_fnc == 'adam':
			opt = keras.optimizers.Adam(learning_rate=learning_rate)
		elif opt_fnc == 'sgd':
			opt = keras.optimizers.SGD(learning_rate=learning_rate)
		else:
			raise ValueError("opt_func must be either \'adam\' or \'sgd\'")

		model.compile(
			optimizer=opt,
			loss=keras.losses.mean_squared_error,      
			# make sure to normalize all outputs, otherwise DCIR values will drastically skew MSE reading compared to error of predicted SOH
			metrics=['accuracy'] )
		return model

	dir_hyperparam_results = dir_repo_main.joinpath("results", 'hyperparameter_optimization')
	plot_params = {
		'ridge': {'legend_key':'Ridge', 'color':'C0',},
		'lasso': {'legend_key':'Lasso', 'color':'C1',},
		'elasticnet': {'legend_key':'ElasticNet', 'color':'C2',},
		'randomforest': {'legend_key':'RF', 'color':'C4',},
		'mlp_chg':{'legend_key':'MLP', 'color':'C5',},
	}

	#region: prepare data for plotting
	#region: create modeling dataset (charge pulse, all SOCs)
	# load pulse and cc data
	cc_data = load_processed_data(data_type='cc')
	pulse_data = load_processed_data(data_type='slowpulse')
	# create modeling data
	all_data = deepcopy(pulse_data)
	idxs = np.where((all_data['pulse_type'] == 'chg'))
	for k in all_data.keys():
		all_data[k] = all_data[k][idxs]
	all_data['voltage_rel'] = np.asarray([v - v[0] for v in all_data['voltage']])
	modeling_data = create_modeling_data(all_data=all_data, input_feature_keys=['voltage_rel'])
	X = modeling_data['model_input']
	y = modeling_data['model_output'][:,0].reshape(-1,1)
	#endregion

	#region: get CV splits
	cv_splitter = Custom_CVSplitter(n_splits=3, split_type='group_id', rand_seed=random_state)
	cv_splits = list(cv_splitter.split(
		X = modeling_data['model_input'], 
		y = modeling_data['model_output'][:,0], 
		cell_ids = modeling_data['cell_id']))
	#endregion
	#endregion

	fig, ax = plt.subplots(figsize=(4,2))
	labels = []
	for i, model_type in enumerate(list(plot_params.keys())):
		file_study = list(dir_hyperparam_results.glob(f'*{model_type}_onlySOH.pkl*'))[0]
		study = pickle.load(open(file_study, 'rb'))
		# create optimal model from saved parameters values
		model = None
		if model_type == 'ridge': model = Ridge(**study.best_trial.params)
		elif model_type == 'lasso': model = Lasso(**study.best_trial.params)
		elif model_type == 'elasticnet': model = ElasticNet(**study.best_trial.params)
		elif model_type == 'randomforest': model = RandomForestRegressor(**study.best_trial.params)
		elif model_type == 'mlp_chg': model = _create_mlp_model(**study.best_trial.params, output_shape=(1,))
		
		# plot avg model error 
		res = _evaluate_model(model, X, y, cv_splits)
		
		ax.bar(i-0.15, res['train'][0], width=0.3, align='center', color='C0')
		ax.bar(i+0.15, res['test'][0], width=0.3, align='center', color='C1')
		# ax.errorbar(i-0.15, res['train'][0], res['train'][1], capsize=2, color='k')
		# ax.errorbar(i+0.15, res['test'][0], res['test'][1], capsize=2, color='k')
		labels.append(plot_params[model_type]['legend_key'])

	ax.set_xlabel("Model Name")
	ax.set_xticks(np.arange(0,len(labels),1), labels=labels, fontsize=8)
	ax.set_ylabel("MAPE [%]")
	ax.set_ylim([0,2.8])
	ax.set_yticks(np.arange(0,2.8,0.5), np.arange(0,2.8,0.5))
	ax.legend(['Train Error', 'Test Error'], ncols=2, loc='upper center', fontsize=8)
	if dir_save is not None:
		filename = dir_save.joinpath("ModelPerformanceBaseline_Pulse.pdf")
		fig.savefig(filename, dpi=300, bbox_inches='tight')
	plt.show()


def plot_optimal_NN(save=False, use_log_norm=True, bias=0.8):
    fontsize_legend = 6
    fontsize_ticklabel = 8
    fontsize_xylabel = 9

    f = dir_results.joinpath("saved_models", "LFP", "slowpulse/chg/2024-10-18/NN_Size_Opt_Gridsearch_Results.pkl")
    gs_results = [pickle.load(open(f, 'rb'))] #[pickle.load(open(f, 'rb')) for f in f_gs_results]

    #region: process raw gridsearch data
    layers = []
    neurons = []
    flops = []
    mapes = []
    for gs_run in gs_results[0].keys():
        layers.append( gs_results[0][gs_run]['params']['n_hlayers'] )
        neurons.append( gs_results[0][gs_run]['params']['n_neurons'] )
        flops.append( gs_results[0][gs_run]['efficiency']['flops (e9)'] * (10**9) )
        cv_keys = list(gs_results[0][gs_run].keys())
        cv_keys.remove('params')
        cv_keys.remove('efficiency')
        test_errors = []
        for cv_key in cv_keys:
            for gs_res in gs_results:
                test_errors.append( gs_res[gs_run][cv_key]['test_error'][0] )
        mapes.append( np.average(test_errors, axis=0) )
    layers = np.array(layers)
    neurons = np.array(neurons)
    flops = np.array(flops)
    mapes = np.array(mapes)
    #endregion

    #region: make grid of mape & flop values at corresponding combination of neurons and layers
    neurons_uniq = np.unique(neurons)
    neuron_dim = len(neurons_uniq)
    layers_uniq = np.unique(layers)
    layer_dim = len(layers_uniq)

    mape_grid = np.zeros( shape=(neuron_dim, layer_dim) )
    flop_grid = np.zeros( shape=(neuron_dim, layer_dim) )
    flop_grid_log = np.zeros( shape=(neuron_dim, layer_dim) )

    for i in range(neuron_dim*layer_dim):
        col = int(i % layer_dim)
        row = int(i / layer_dim)
        idx = np.where((neurons==neurons_uniq[row]) & (layers==layers_uniq[col]))[0][0]
        mape_grid[row][col] = mapes[idx,0]
        flop_grid[row][col] = flops[idx]
        flop_grid_log[row][col] = np.log(flop_grid[row][col])
    #endregion
    
    #region: create normalized grids of MAPE & FLOPS
    mape_grid_norm = np.zeros(mape_grid.shape)
    flop_grid_norm = np.zeros(flop_grid.shape)
    flop_grid_lognorm = np.zeros(flop_grid.shape)

    for r in range(mape_grid.shape[0]):
        for c in range(mape_grid.shape[1]):
            mape_grid_norm[r][c] = (mape_grid[r][c] - np.min(mape_grid)) / (np.max(mape_grid) - np.min(mape_grid))
            flop_grid_norm[r][c] = (flop_grid[r][c] - np.min(flop_grid)) / (np.max(flop_grid) - np.min(flop_grid))
            flop_grid_lognorm[r][c] = (flop_grid_log[r][c] - np.min(flop_grid_log)) / (np.max(flop_grid_log) - np.min(flop_grid_log))
    #endregion

    #region: compute best NN size
    def opt_func(mape, flop, weight=0.5):
        return np.sqrt( (weight*mape)**2 + ((1-weight)*flop)**2 )
    
    opt_grid = np.zeros( shape=(neuron_dim, layer_dim) )
    best_params = {'neurons':None, 'layers':None, 'mape':None, 'flops':None, 'opt_fnc':None}
    for r in range(mape_grid.shape[0]):
        for c in range(mape_grid.shape[1]):
            opt_val = None
            # if use_log_norm:
            #     opt_val = opt_func(mape_grid_norm[r][c], flop_grid_lognorm[r][c], weight=bias)
            # else:
            opt_val = opt_func(mape_grid_norm[r][c], flop_grid_norm[r][c], weight=bias)
            opt_grid[r][c] = opt_val
            if best_params['opt_fnc'] is None or opt_val < best_params['opt_fnc']:
                best_params['neurons'] = neurons_uniq[r]
                best_params['layers'] = layers_uniq[c]
                best_params['mape'] = mape_grid[r][c]
                best_params['flops'] = flop_grid[r][c]
                best_params['opt_fnc'] = opt_val
    #endregion  
    
    #region: Figure - Heat map of MAPE & FLOPS v NN size
    fig = plt.figure(figsize=(6.25,2), constrained_layout=True)
    gs = GridSpec(1, 6, figure=fig, width_ratios=[8,1,8,1,8,1])
    axes = [
        fig.add_subplot(gs[0]), # neurons v. layers grid
        fig.add_subplot(gs[1]), # MAPE colorbar
        fig.add_subplot(gs[2]), # neurons v. layers grid
        fig.add_subplot(gs[3]),  # FLOPS colorbar
        fig.add_subplot(gs[4]), # neurons v. layers grid
        fig.add_subplot(gs[5])  # opt_func colorbar
    ]

    mape_norm = mpl.colors.Normalize(vmin=np.min(mape_grid), vmax=np.max(mape_grid))
    cm_mape = mpl.cm.Blues
    scm_mape = mpl.cm.ScalarMappable(cmap=cm_mape, norm=mape_norm)
    scm_mape.set_array([])

    flop_norm = None
    if use_log_norm:
        flop_norm = mpl.colors.LogNorm(vmin=np.min(flop_grid), vmax=np.max(flop_grid))
    else:
        flop_norm = mpl.colors.Normalize(vmin=np.min(flop_grid), vmax=np.max(flop_grid))
    cm_flop = mpl.cm.Reds
    scm_flop = mpl.cm.ScalarMappable(cmap=cm_flop, norm=flop_norm)
    scm_flop.set_array([])
    
    opt_norm = None
    if use_log_norm:
        opt_norm = mpl.colors.LogNorm(vmin=np.min(opt_grid), vmax=np.max(opt_grid))
    else:
        opt_norm = mpl.colors.Normalize(vmin=np.min(opt_grid), vmax=np.max(opt_grid))
    cm_opt = mpl.cm.Purples
    scm_opt = mpl.cm.ScalarMappable(cmap=cm_opt, norm=opt_norm)
    scm_opt.set_array([])

    axes[0].imshow(mape_grid, cmap=cm_mape, norm=mape_norm, interpolation='nearest', 
                aspect='equal', origin='lower')
    axes[0].set_title("MAPE of $Q_{dchg}$", fontsize=fontsize_xylabel)
    axes[0].set_xlabel("Hidden Layers [-]", fontsize=fontsize_xylabel)
    axes[0].set_ylabel("Neurons [-]", fontsize=fontsize_xylabel)
    axes[0].set_xticks(np.arange(0, len(layers_uniq), 1))
    axes[0].set_xticklabels(layers_uniq, fontsize=fontsize_ticklabel)
    axes[0].set_yticks(np.arange(0,neuron_dim,1))
    axes[0].set_yticklabels(neurons_uniq, fontsize=fontsize_ticklabel)
    cbar = fig.colorbar(scm_mape, cax=axes[1], 
                        ticks=np.arange(0, int(np.max(mape_grid)+1),0.2))
    cbar.ax.set_title('[%]', fontsize=fontsize_ticklabel)
    cbar.ax.tick_params(labelsize=fontsize_ticklabel)

    axes[2].imshow(flop_grid, cmap=cm_flop, norm=flop_norm, interpolation='nearest', 
                aspect='equal', origin='lower')
    axes[2].set_title("FLOPs", fontsize=fontsize_xylabel)
    axes[2].set_xlabel("Hidden Layers [-]", fontsize=fontsize_xylabel)
    axes[2].set_ylabel("Neurons [-]", fontsize=fontsize_xylabel)
    axes[2].set_xticks(np.arange(0, len(layers_uniq), 1))
    axes[2].set_xticklabels(layers_uniq, fontsize=fontsize_ticklabel)
    axes[2].set_yticks(np.arange(0,neuron_dim,1))
    axes[2].set_yticklabels(neurons_uniq, fontsize=fontsize_ticklabel)
    cbar = fig.colorbar(scm_flop, cax=axes[3])
    cbar.ax.set_title('[-]', fontsize=fontsize_ticklabel)
    cbar.ax.tick_params(labelsize=fontsize_ticklabel)
    
    axes[4].imshow(opt_grid, cmap=cm_opt, norm=opt_norm, interpolation='nearest', 
                aspect='equal', origin='lower')
    axes[4].set_title("Optimal Model", fontsize=fontsize_xylabel)
    axes[4].set_xlabel("Hidden Layers [-]", fontsize=fontsize_xylabel)
    axes[4].set_ylabel("Neurons [-]", fontsize=fontsize_xylabel)
    axes[4].set_xticks(np.arange(0, len(layers_uniq), 1))
    axes[4].set_xticklabels(layers_uniq, fontsize=fontsize_ticklabel)
    axes[4].set_yticks(np.arange(0,neuron_dim,1))
    axes[4].set_yticklabels(neurons_uniq, fontsize=fontsize_ticklabel)
    cbar = fig.colorbar(scm_opt, cax=axes[5],
                        ticks=[np.min(opt_grid), np.max(opt_grid)])
    cbar.ax.minorticks_off()
    cbar.ax.set_yticklabels(['Best', 'Worst'], fontsize=fontsize_ticklabel)
    if save: 
        dir_save = dir_figures.joinpath("raw", "NN Size Optimization")
        dir_save.mkdir(parents=True, exist_ok=True)
        filename = "NN Size Optimization - {} bias={}.png".format("logNorm" if use_log_norm else "Norm", bias)
        plt.savefig(dir_save.joinpath(filename), dpi=300)
        print(f"Figure saved to: {dir_save.joinpath(filename)}")
    plt.show()
    
    print("The optimal NN structure is: ")
    print(best_params)
    #endregion

def plot_aging_trajectories(plot_all_trajectories:bool=False, extrapolate:bool=False, soh_bounds:tuple=(80,100), save:bool=False):
	"""Plots the aging trajectories of the LFP/Gr dataset.

	Args:
		plot_all_trajectories (bool, optional): If True, the trajectory of each cell is plotted. Else, the mean trajectory of each cycling group is plotted. Defaults to False.
		extrapolate (bool, optional): Whether to linearly extend the aging trajectories to future SOHs. Data will be extended to the lower limit of the 'soh_bounds' parameter. Defaults to False.
		soh_bounds (tuple, optional): The SOH bounds to plot between. Defaults to (80,100).
		save (bool, optional). Whether to save the figure. If True, the path location of the saved plot is printed. Defaults to False

		"""
	from scipy.stats import norm
	all_data = load_processed_data(data_type='slowpulse')
	data = {'Cell ID':[], 'Group ID':[], 'RPT':[], 'Num Cycles':[], 'Capacity (Ah)':[], 'SOH (%)':[]}

	#region: calculate required features
	for cell_id in np.unique(all_data['cell_id']):
		filt_idxs = np.where((all_data['cell_id'] == cell_id) & (all_data['pulse_type'] == 'chg') & (all_data['soc'] == 20))
		sort_idxs = np.argsort(all_data['rpt'][filt_idxs])

		rpts = all_data['rpt'][filt_idxs][sort_idxs]
		cycles = all_data['num_cycles'][filt_idxs][sort_idxs]
		# cycles may be NaN (missing cycling data for that week)
		# replace with average of cycles from previous and next weeks
		for idx in np.where(np.isnan(cycles))[0]:		
			lb = cycles[idx-1] if idx > 1 else None
			ub = cycles[idx+1] if idx < len(cycles) else None
			if lb is None and ub is None: cycles[idx] = 0
			elif lb is None: cycles[idx] = ub
			else: cycles[idx] = (lb + ub) / 2
		q_dchgs = all_data['q_dchg'][filt_idxs][sort_idxs]
		sohs = q_dchgs / q_dchgs[0] * 100

		assert len(np.where(np.isnan(cycles))[0]) == 0
		assert len(rpts) == len(cycles) == len(q_dchgs) == len(sohs)

		data['Cell ID'].append(np.full_like(rpts, cell_id))
		data['Group ID'].append(np.full_like(rpts, get_group_id(cell_id=cell_id)))
		data['RPT'].append(rpts)
		data['Num Cycles'].append(np.cumsum(cycles))
		data['Capacity (Ah)'].append(q_dchgs)
		data['SOH (%)'].append(sohs)

	data['Cell ID'] = np.hstack(data['Cell ID'])
	data['Group ID'] = np.hstack(data['Group ID'])
	data['RPT'] = np.hstack(data['RPT'])
	data['Num Cycles'] = np.hstack(data['Num Cycles'])
	data['Capacity (Ah)'] = np.hstack(data['Capacity (Ah)'])
	data['SOH (%)'] = np.hstack(data['SOH (%)'])
	#endregion

	#region: interpolate data to fixed SOH increments
	cols_for_interp = ['Cell ID', 'Group ID', 'Num Cycles', 'Capacity (Ah)', 'SOH (%)']
	data_interp = {k:[] for k in cols_for_interp}
	sohs_interp = np.arange(100, 75, -1)
	for cell_id in np.unique(data['Cell ID']):
		filt_idxs = np.where(data['Cell ID'] == cell_id)
		data_interp['SOH (%)'].append(sohs_interp)
		data_interp['Cell ID'].append( np.full(len(sohs_interp), cell_id) )
		data_interp['Group ID'].append( np.full(len(sohs_interp), data['Group ID'][filt_idxs][0]) )

		for column in ['Num Cycles', 'Capacity (Ah)']:
			#region: interpolate feature to pre-defined SOH increments
			sohs = data['SOH (%)'][filt_idxs]
			vals = data[column][filt_idxs]

			interp_success = False
			f = None
			while not interp_success:
				try:
					f = interpolate.PchipInterpolator(np.flip(sohs), np.flip(vals), extrapolate=False)
					interp_success = True
				except ValueError:
					for i in range(1, len(sohs)):
						if sohs[i-1] <= sohs[i]: 
							sohs[i-1] = sohs[i]+0.000001
			vals_interp = f(sohs_interp)
			#endregion
			
			#region: linearly extrapolate cells for lower SOHs
			if extrapolate:
				n_samples_to_fit = 2
				last_three_idxs = np.zeros(n_samples_to_fit)       # last n idxs on non-nan values
				for i in range(len(vals_interp)):
					if not np.isnan(vals_interp[i]):
						last_three_idxs[i%n_samples_to_fit] = i
				last_three_idxs = np.sort(last_three_idxs).astype(int)

				p = np.poly1d(np.polyfit(sohs_interp[last_three_idxs], vals_interp[last_three_idxs], deg=1))
				vals_interp[np.max(last_three_idxs)+1:] = p(sohs_interp[np.max(last_three_idxs)+1:])
			#endregion
			
			data_interp[column].append(vals_interp)

	#region: format interpolated dataframe for plotting
	data_interp = {k:np.asarray(v).ravel() for k,v in data_interp.items()}
	df_plotting_interp = pd.DataFrame(data_interp)
	df_plotting_interp['Group ID'] = df_plotting_interp['Group ID'].astype(int)
	#endregion
	#endregion

	#region: define colors for each group
	colors = [
		'#4E79A7', '#A0CBE8', 
		'#F28E2B', '#FFBE7D',
		'#59A14F', '#8CD17D',
		'#D37295', '#FABFD2',
		'#499894', '#86BCB6',
		'#E15759', '#FF9D9A',
		'#363433', '#94908e',
		'#B6992D', '#F1CE63',
		'#B07AA1', '#D4A6C8',
		'#9D7660', '#D7B5A6',
		'#637939', '#8CA252',
	]
	color_bins = np.arange(0,22,1)
	assert len(color_bins)== len(colors)
	cmap = mpl.colors.ListedColormap(colors)
	cnorm = mpl.colors.Normalize(vmin=0, vmax=22)
	sc_map = mpl.cm.ScalarMappable(cmap=cmap, norm=cnorm)
	sc_map.set_array([])
	#endregion

	#region: Plot aging trajectory of each group w/ group distribution below X axis
	fig = plt.figure(figsize=(4,4))
	gs = GridSpec(figure=fig, nrows=2, ncols=1, height_ratios=[3,1])
	axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
	axes[1].sharex(axes[0])
	max_num_cycles = int(np.ceil(np.max(data['Num Cycles']) / 500) * 500)

	for group_id in df_plotting_interp['Group ID'].dropna(inplace=False).unique():
		group_id = int(group_id)
		df_group = df_plotting_interp.loc[df_plotting_interp['Group ID'] == group_id]
		color_idx = int(group_id)-1
		
		# get avg and std of trajectory for each group
		df_avg = df_group.groupby('SOH (%)', dropna=True).mean().reset_index()
		df_std = df_group.groupby('SOH (%)', dropna=True).std().reset_index()
		
		# plot all cell trajectories
		if plot_all_trajectories:
			add_label = True
			for cell_id in df_group['Cell ID'].unique():
				df_cell = df_group.loc[df_group['Cell ID'] == cell_id]
				if add_label:
					axes[0].plot(df_cell['Num Cycles'].values, df_cell['SOH (%)'].values,
								'-', c=colors[2*color_idx], label=group_id)
					add_label = False
				else:
					axes[0].plot(df_cell['Num Cycles'].values, df_cell['SOH (%)'].values,
								'-', c=colors[2*color_idx])
		
		# plot only mean trajectory
		else:
			axes[0].plot(df_avg['Num Cycles'].values, df_avg['SOH (%)'].values,
						'-', linewidth=3, c=colors[2*color_idx], label=group_id)
		
		#region: plot distributions
		def find_closest_idx(value, df, colname='SOH (%)'):
			exactmatch = df.loc[df[colname] == value]
			if not exactmatch.empty:
				return exactmatch.index
			elif df[colname].min() > soh_bounds[0]:
				return df[colname].idxmin()
			else:
				lower_idx = df.loc[df[colname] < value][colname].idxmin()
				upper_idx = df.loc[df[colname] > value][colname].idxmax()
				lower_val = df[colname][lower_idx]
				upper_val = df[colname][upper_idx]
				if (value - lower_val) / (upper_val - lower_val) > 0.5:
					return upper_idx
				else:
					return lower_idx
		
		distr_idx = find_closest_idx(soh_bounds[0], df_avg, colname='SOH (%)')
		xvals = np.linspace(df_avg['Num Cycles'][distr_idx] - 3*df_std['Num Cycles'][distr_idx],
							df_avg['Num Cycles'][distr_idx] + 3*df_std['Num Cycles'][distr_idx],
							100)
		
		axes[1].plot(xvals, 
					norm.pdf(xvals, 
							loc=df_avg['Num Cycles'][distr_idx], 
							scale=df_std['Num Cycles'][distr_idx])*100,
					c=colors[2*color_idx])
		axes[1].fill(xvals, 
					norm.pdf(xvals, 
							loc=df_avg['Num Cycles'][distr_idx], 
							scale=df_std['Num Cycles'][distr_idx])*100,
					alpha=0.5,
					c=colors[2*color_idx+1])
		#endregion

	#region: set title and labels
	axes[0].legend(title='Group ID', ncol=6, fontsize=8,
					loc='upper center', bbox_to_anchor=(0.5, 1.6), handlelength=1, borderpad=0.75,
					fancybox=True, shadow=True)
	axes[0].set_ylim(soh_bounds)
	axes[0].set_xlim(0, max_num_cycles)
	axes[0].set_ylabel("SOH [%]")
	axes[1].set_xlabel("Number of Cycles [-]")
	axes[1].set_ylabel("PDF")
	axes[1].set_yticks([0,1.0,2.0])
	plt.setp(axes[0].get_xticklabels(), visible=False)

	# axes[0].grid(which='major', color='#AAAAAA', linestyle='-', linewidth=0.5)
	# axes[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
	# axes[0].minorticks_on()

	# axes[1].grid(which='major', color='#AAAAAA', linestyle='-', linewidth=0.5)
	# axes[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.5)
	# axes[1].minorticks_on()
	#endregion

	fig.align_labels()
	fig.tight_layout(pad=0.8)
	filename = "Aging Trajectories - Mean with PDF.png"
	# if save: fig.savefig(f_save_plot_folder.joinpath("Dataset Analysis", filename), dpi=300)#, bbox_inches="tight")
	plt.show()
	#endregion

	if save: 
		dir_save = dir_figures.joinpath("raw", "Aging Trajectory")
		dir_save.mkdir(parents=True, exist_ok=True)
		filename = f"LFP Aging Trajectory - {'GroupMean' if not plot_all_trajectories else 'AllCells'}{'-Extrap' if extrapolate else ''}-{soh_bounds}.png"
		plt.savefig(dir_save.joinpath(filename), dpi=300)
		print(f"Figure saved to: {dir_save.joinpath(filename)}")
	plt.show()




if __name__ == '__main__':
    print("plotting.py")

