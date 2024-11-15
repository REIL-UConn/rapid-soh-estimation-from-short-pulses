
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from common_methods import *

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

rand_seed = 1234



def get_pulse_model_dataset(chemistry:str, pulse_type:str, pulse_soc=None, pulse_len:int=100):
	# TODO: pulse_len currently does nothing. It is intended to be used to interpolate the pulse to different lengths
	assert chemistry in ['LFP', 'NMC']
	assert pulse_type in ['chg', 'dchg']  	# TODO: add 'both' option to combine chg and dchg pulses
	if pulse_soc is not None: assert pulse_soc in ([20,50,90] if chemistry == 'LFP' else [30,50,70])

	if chemistry == 'LFP':
		pulse_data = load_processed_data(data_type='slowpulse')
		filt_idxs = None
		if pulse_soc is None:
			filt_idxs = np.where(pulse_data['pulse_type'] == pulse_type)
		else:
			filt_idxs = np.where((pulse_data['pulse_type'] == pulse_type) & (pulse_data['soc'] == pulse_soc))

		output_keys = ['q_dchg', 'dcir_chg_20', 'dcir_chg_50', 'dcir_chg_90', 'dcir_dchg_20', 'dcir_dchg_50', 'dcir_dchg_90']
		modeling_dic = {
			'model_input':np.asarray([v-v[0] for v in pulse_data['voltage'][filt_idxs]]),
			'model_output':np.vstack([pulse_data[k][filt_idxs] for k in output_keys]).T,
			'model_cell_id':pulse_data['cell_id'][filt_idxs],
			'model_group_id':pulse_data['group_id'][filt_idxs],
		}

		#region: create scalers and scaled data
		modeling_dic['model_input_scaler'] = StandardScaler()
		modeling_dic['model_input_scaled'] = modeling_dic['model_input_scaler'].fit_transform(modeling_dic['model_input'])
		modeling_dic['model_output_scaler'] = StandardScaler()
		modeling_dic['model_output_scaled'] = modeling_dic['model_output_scaler'].fit_transform(modeling_dic['model_output'])
		#endregion

		#region format arrays
		modeling_dic['model_output'] = np.array(modeling_dic['model_output'])
		modeling_dic['model_input'] = np.array(modeling_dic['model_input'])
		modeling_dic['model_cell_id'] = np.array(modeling_dic['model_cell_id'])
		modeling_dic['model_group_id'] = np.array(modeling_dic['model_group_id'])
		#endregion

		#region: assert all arrays are same length (same number of samples)
		temp = len(modeling_dic['model_cell_id'])
		for k in modeling_dic.keys():
			if hasattr(modeling_dic[k], '__len__'): assert len(modeling_dic[k]) == temp
		#endregion

		return modeling_dic

	elif chemistry == 'NMC':
		from postprocessing_NMC import get_dataset_for_pulse_modeling
		modeling_dic = get_dataset_for_pulse_modeling(
			pulse_type=pulse_type,
			pulse_soc=pulse_soc,
			c_rate=1, 
			use_relative_voltage=True, 
			remove_CV=False
		)
		return modeling_dic


if __name__ == '__main__':
	# model input type / parameters
	chemistry = 'LFP'
	pulse_type = 'chg'  # {'chg', 'dchg'}
	pulse_soc = None    # {20, 50, 90, None}. None will use pulses from all SOCs
	input_size = 100    # pulse will be interpolated to this size (default is 100)

	#region: model optimization parameters
	perform_grid_search = True      # whether to run new grid search (also saves new model using best results)
	gridsearch_params = {
		'n_hlayers': [5],           
		'n_neurons': [100],          
		'act_fnc': ['tanh'],   			# 'linear','relu','sigmoid','softmax','softplus',

		'opt_fnc': ['adam'],  			# ['adam', 'sgd'],
		'learning_rate': [0.0015],      # [.005, .001, .0015],
		
		'batch_size': [50],             # [25,50,75,100],
		'epochs': [500], 
	}
	num_cv_splits = 3
	#endregion

	#region: model saving parameters / paths
	f_saved_models_folder = dir_results.joinpath("saved_models", chemistry, "slowpulse", pulse_type.lower())
	today_date_str = datetime.today().strftime("%Y-%m-%d")
	f_save_results_path = f_saved_models_folder.joinpath(today_date_str)
	f_save_results_path.parent.mkdir(parents=True, exist_ok=True)
	#endregion

	# get dataset for this model type
	model_dataset = get_pulse_model_dataset(chemistry='LFP', pulse_type=pulse_type, pulse_soc=pulse_soc, pulse_len=input_size)

	#region: if configured, perform a gridsearch and save all & best results
	if perform_grid_search:
		print("Performing gridsearch...")
		
		gridsearch_results = run_custom_gridsearch(
			model_dataset, gridsearch_params, n_cv_splits=num_cv_splits, verbose=True, record_flops=False,
			split_type=('group_id' if chemistry == 'LFP' else 'cell_id'))
		best_run = get_best_gridsearch_run(gridsearch_results)

		#region: Save all gridsearch results
		f_saved_grid_result_all = f_save_results_path.joinpath("Gridsearch_Results_All.pkl")
		# increment filename if already exists
		idx = 1
		while f_saved_grid_result_all.exists():
			f_saved_grid_result_all = f_save_results_path.joinpath("Gridsearch_Results_All_{}.pkl".format(idx))
			idx += 1
		f_saved_grid_result_all.parent.mkdir(exist_ok=True, parents=True)
		with open(f_saved_grid_result_all,'wb') as file:
			pickle.dump(gridsearch_results, file)
		file.close()
		#endregion

		#region: Save best gridsearch results
		f_saved_grid_result_best = f_save_results_path.joinpath("Gridsearch_Results_Best.pkl")
		# increment filename if already exists
		idx = 1
		while f_saved_grid_result_best.exists():
			f_saved_grid_result_best = f_save_results_path.joinpath("Gridsearch_Results_Best_{}.pkl".format(idx))
			idx += 1
		with open(f_saved_grid_result_best,'wb') as file:
			pickle.dump(best_run, file)
		file.close()
		print('Best params: ', best_run['params'] )
		#endregion
		
		#region: Train new model using best params
		print("Training new model using best parameters from gridsearch results")
		# get latest saved gridsearch results
		f_all_best_grid_results = sorted(f_save_results_path.glob("Gridsearch_Results_Best*"))
		f_best_results = f_all_best_grid_results[-1]
		best_results = pickle.load(open(f_best_results, 'rb'))

		# define early stop callback for CV training
		early_stop = keras.callbacks.EarlyStopping(
			monitor='loss', min_delta=0, patience=5, verbose=False, 
			mode='auto', baseline=None, restore_best_weights=True)
		# generate cross validation splits
		cvSplitter = Custom_CVSplitter(n_splits=num_cv_splits, split_type=('group_id' if chemistry == 'LFP' else 'cell_id'))
		cv_splits = cvSplitter.split(
			model_dataset['model_input_scaled'], 
			model_dataset['model_output_scaled'], 
			model_dataset['model_cell_id']) 
		cv_splits = list(cv_splits)

		# create model with these best parameters
		fit_params = {}
		if 'epochs' in best_results['params'].keys():
			fit_params['epochs'] = best_results['params']['epochs']
			best_results['params'].pop('epochs')
		if 'batch_size' in best_results['params'].keys():
			fit_params['batch_size'] = best_results['params']['batch_size']
			best_results['params'].pop('batch_size')
		model = create_model(**best_results['params'], input_size=len(model_dataset['model_input'][0]))

		# evaluate each cross-validation fold
		results = {}
		cv_idx = 0
		print("Performing {}-fold Cross Validation:".format(num_cv_splits))
		for train_idxs, test_idxs in cv_splits:
			print("  CV fold {}".format(cv_idx))
			
			model.fit(model_dataset['model_input_scaled'][train_idxs], 
						model_dataset['model_output_scaled'][train_idxs],
						**fit_params,
						callbacks=early_stop,
						verbose=False )

			# get test accuracy
			y_pred_scaled = model.predict( model_dataset['model_input_scaled'][test_idxs], verbose=False)
			y_pred = model_dataset['model_output_scaler'].inverse_transform(y_pred_scaled)
			test_error = get_prediction_error( model_dataset['model_output'][test_idxs], y_pred )
			
			# get train accuracy
			X_pred_scaled = model.predict( model_dataset['model_input_scaled'][train_idxs], verbose=False )
			X_pred = model_dataset['model_output_scaler'].inverse_transform(X_pred_scaled)
			train_error = get_prediction_error( model_dataset['model_output'][train_idxs], X_pred )
			
			# record test and train error in gridsearch_results
			results['CV {}'.format(cv_idx)] = {
				'train_error': train_error,
				'test_error': test_error
			}
			cv_idx += 1
		#endregion
		
		#region: save model & scalers
		f_saved_model = f_save_results_path.joinpath("Model_KerasSeq.h5")
		# increment filename if already exists
		idx = 1
		while f_saved_model.exists():
			f_saved_model = f_save_results_path.joinpath("Model_KerasSeq_{}.h5".format(idx))
			idx += 1
		model.save(f_saved_model, save_format='h5')

		# save input scaler
		f_saved_scaler_input = f_save_results_path.joinpath("Model_Scaler_Input.pkl")
		# increment filename if already exists
		idx = 1
		while f_saved_scaler_input.exists():
			f_saved_scaler_input = f_save_results_path.joinpath("Model_Scaler_Input_{}.pkl".format(idx))
			idx += 1
		with open(f_saved_scaler_input,'wb') as file:
			pickle.dump(model_dataset['model_input_scaler'], file)
		file.close()

		# save ouput scaler
		f_saved_scaler_output = f_save_results_path.joinpath("Model_Scaler_Output.pkl")
		# increment filename if already exists
		idx = 1
		while f_saved_scaler_output.exists():
			f_saved_scaler_output = f_save_results_path.joinpath("Model_Scaler_Output_{}.pkl".format(idx))
			idx += 1
		with open(f_saved_scaler_output,'wb') as file:
			pickle.dump(model_dataset['model_output_scaler'], file)
		file.close()
		
		print("Model and input/scalers saved to \'Saved Model\' folder")
		print("Gridsearch complete.")
		#endregion
	#endregion  
		
	#region: load most recent model if not performing a new gridsearch
	print("Testing best saved model...")
	# get latest saved model and scalers
	f_latest_models_folder = get_latest_models_folder(f_saved_models_folder, "*.h5")
	f_all_saved_models = sorted( f_latest_models_folder.glob("*.h5") )
	f_best_model = f_all_saved_models[-1] 
	f_all_input_scalers = sorted( f_latest_models_folder.glob("Model_Scaler_Input*") )
	f_best_input_scaler = f_all_input_scalers[-1]
	f_all_output_scalers = sorted( f_latest_models_folder.glob("Model_Scaler_Output*") )
	f_best_output_scaler = f_all_output_scalers[-1]

	# load model and scaler
	input_scaler = pickle.load(open(f_best_input_scaler, 'rb'))
	output_scaler = pickle.load(open(f_best_output_scaler, 'rb'))
	model = tf.keras.models.load_model(f_best_model)

	# perform prediction
	model_dataset['model_input_scaler'] = input_scaler
	model_dataset['model_output_scaler'] = output_scaler
	pred_results = perform_prediction(model, model_dataset, num_cv_splits, rand_seed)

	# display [num_cv_splits]-fold average prediction accuracy
	print("{}-Fold CV Accuracy".format(num_cv_splits))
	features = ["Q", "R_CHG_20", "R_CHG_50", "R_CHG_90", "R_DCHG_20", "R_DCHG_50", "R_DCHG_90"]
	units = ["Ahr", "Ohm", "Ohm", "Ohm", "Ohm", "Ohm", "Ohm"]
	for feature_idx in range(7):
		print("  {}".format(features[feature_idx]))
		print("    - MAPE: mean={}%,  std={}%".format( round(pred_results['test']['MAPE'][0][feature_idx], 6), 
														round(pred_results['test']['MAPE'][1][feature_idx], 6)  ))
		print("    - RMSE: mean={} {},  std={} {}".format( round(pred_results['test']['RMSE'][0][feature_idx], 6), 
															units[feature_idx], 
															round(pred_results['test']['RMSE'][1][feature_idx], 6),
															units[feature_idx]  ))

	# plot results
	f_save_plots_path = dir_figures.joinpath("raw", "_slowpulse_py_", chemistry, f"pulse_type={pulse_type}")
	f_save_plots_path.mkdir(parents=True, exist_ok=True)
	plot_predictions(pred_results, "Slow Pulse", save_fig=True, save_path=f_save_plots_path)
	#endregion

	print(model.summary())
	
	print("slowpulse.py complete.")




	
