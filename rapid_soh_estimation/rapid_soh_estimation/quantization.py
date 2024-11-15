
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from common_methods import *

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

from slowpulse import get_pulse_model_dataset

class QUANTIZATION_LEVEL(Enum):
    INT8 = 1
    FLOAT16 = 2
    # DYNAMIC = 3         # NOT IMPLEMENTED YET

rand_seed = 1234    # set to None if want truly random plot, set value if want reproducible plots


# Converts TFL model to a C++ header files
def tfl_to_hex_array(tf_model, tfl_model, f_tfl_model, scalers=(None,None)):
	c_str = ''

	#  add header guard
	header_str = f_tfl_model.name[:f_tfl_model.name.rfind('.')]
	c_str += '#ifndef ' + header_str.upper() + '_H\n'
	c_str += '#define ' + header_str.upper() + '_H\n\n'
	
	#  add includes
	c_str += '#include <Arduino.h>\n\n'
	
	#  add model requirements as definitions
	c_str += '#define INPUT_SIZE\t' + str(tf_model.input_shape[1])
	c_str += '\n'
	c_str += '#define OUTPUT_SIZE\t' + str(tf_model.output_shape[1])
	c_str += '\n\n'

	#region: add StandardScaler values so can appropriately transform input and inverse transform output
	if scalers[0] is not None:
		scaler_X_means = scalers[0].mean_
		scaler_X_stds = scalers[0].scale_
		# X scaler means
		c_str += 'PROGMEM const float ' + 'xScalerMeans' + '[] = {'
		scaler_array = []
		for i, val in enumerate(scaler_X_means):
			float_str = str(val)
			if (i+1) < len(scaler_X_means):
				float_str += ','
			if (i+1)%4 == 0:
				float_str += '\n '
			scaler_array.append(float_str)
		# Add closing brace
		c_str += '\n  ' + format(' '.join(scaler_array)) + '\n};\n\n'
		# X scaler stds
		c_str += 'PROGMEM const float ' + 'xScalerSTDs' + '[] = {'
		scaler_array = []
		for i, val in enumerate(scaler_X_stds):
			float_str = str(val)
			if (i+1) < len(scaler_X_stds):
				float_str += ','
			if (i+1)%4 == 0:
				float_str += '\n '
			scaler_array.append(float_str)
		# Add closing brace
		c_str += '\n  ' + format(' '.join(scaler_array)) + '\n};\n\n'
	if scalers[1] is not None:
		scaler_y_means = scalers[1].mean_
		scaler_y_stds = scalers[1].scale_
		# y scaler means
		c_str += 'PROGMEM const float ' + 'yScalerMeans' + '[] = {'
		scaler_array = []
		for i, val in enumerate(scaler_y_means):
			float_str = str(val)
			if (i+1) < len(scaler_y_means):
				float_str += ','
			if (i+1)%4 == 0:
				float_str += '\n '
			scaler_array.append(float_str)
		# Add closing brace
		c_str += '\n  ' + format(' '.join(scaler_array)) + '\n};\n\n'
		# y scaler stds
		c_str += 'PROGMEM const float ' + 'yScalerSTDs' + '[] = {'
		scaler_array = []
		for i, val in enumerate(scaler_y_stds):
			float_str = str(val)
			if (i+1) < len(scaler_y_stds):
				float_str += ','
			if (i+1)%4 == 0:
				float_str += '\n '
			scaler_array.append(float_str)
		# Add closing brace
		c_str += '\n  ' + format(' '.join(scaler_array)) + '\n};\n\n'
	#endregion
	
	#region: Add quantization values
	tfmodel_interp = tf.lite.Interpreter(model_path=str(f_tfl_model))
	qt_input_scale, qt_input_zeropoint = tfmodel_interp.get_input_details()[0]["quantization"]
	qt_output_scale, qt_output_zeropoint = tfmodel_interp.get_output_details()[0]["quantization"]
	
	c_str += 'PROGMEM const double ' + 'quantScalerInput' + ' = ' + str(qt_input_scale) + ';\n'
	c_str += 'PROGMEM const uint8_t ' + 'quantZeropointInput' + ' = ' + str(qt_input_zeropoint) + ';\n'
	c_str += 'PROGMEM const double ' + 'quantScalerOutput' + ' = ' + str(qt_output_scale) + ';\n'
	c_str += 'PROGMEM const uint8_t ' + 'quantZeropointOutput' + ' = ' + str(qt_output_zeropoint) + ';\n'
	#endregion
	
	# add model array length 
	c_str += '\nPROGMEM const unsigned int ' + 'model_pulse_len = ' + str(len(tfl_model)) + ';\n'

	# Declare C variable
	c_str += 'PROGMEM const unsigned char ' +  'model_pulse' + '[] = {'
	hex_array = []
	for i, val in enumerate(tfl_model):
		hex_str = format(val, '#04x')
		if (i+1) < len(tfl_model):
			hex_str += ','
		if (i+1)%12 == 0:
			hex_str += '\n '
		hex_array.append(hex_str)
	# Add closing brace
	c_str += '\n  ' + format(' '.join(hex_array)) + '\n};\n\n'

	# Close header guard
	c_str += '#endif //' + header_str.upper() + '_H'

	return c_str

# simple TFL model wrapper to add .predict function
class TFL_Model_Wrapper():
	def __init__(self, f_tfl_model):
		self.f_tfl_model = f_tfl_model
		# Initialize the interpreter
		self.interpreter = tf.lite.Interpreter(model_path=str(self.f_tfl_model))
		self.interpreter.allocate_tensors()
		
	def predict(self, X, verbose=False):
		input_details = self.interpreter.get_input_details()[0]
		output_details = self.interpreter.get_output_details()[0]

		predictions = []
		for i, input in enumerate(X):
			# check if tfl model is quantized --> if so, quantize input data
			if input_details['dtype'] == np.uint8:
				input_scale, input_zero_point = input_details['quantization']
				input = (input / input_scale) + input_zero_point

			# set input tensor
			input = np.expand_dims(input, axis=0).astype(input_details["dtype"])
			self.interpreter.set_tensor(input_details["index"], input)

			# run inference
			self.interpreter.invoke()
			
			# get outputs
			output = self.interpreter.get_tensor(output_details["index"])[0]
			# check if output needs to be dequantized
			if output_details['dtype'] == np.uint8:
				output_scale, output_zero_point = output_details['quantization']
				output_scale = float(output_scale)
				output_zero_point = float(output_zero_point)
				# dequantize each
				output_dq = np.zeros(len(output))
				for k in range( len(output)):
					output_dq[k] = (float(output[k]) - output_zero_point) * output_scale
				predictions.append(output_dq)
			else:
				predictions.append(output)

		return predictions

def plot_quantization_comparison(model_full, model_tfl, model_dataset, feature='dchg_q', dir_save:Path=None):
	'model_full: provide the trained model (32bit)'
	'model_tfl: provide the trained model (8bit quantized)'
	'model_dataset: provide the dataset to use (updated scalers if loading a saved model)'
	'feature: output featuer to compare'
	'save: optional specifier to save the figure'
	
	all_features = ['dchg_q', 
					'dcir_chg_20soc', 'dcir_chg_50soc', 'dcir_chg_90soc',
					'dcir_dchg_20soc', 'dcir_dchg_50soc', 'dcir_dchg_90soc' ]
	assert feature in all_features, "\'feature\' must be one of the following: {}".format(all_features)
	
	# randomly pick a cv_split to use for testing
	cvSplitter = Custom_CVSplitter(n_splits=num_cv_splits, rand_seed=rand_seed)
	cv_splits = cvSplitter.split(model_dataset['model_input_scaled'], model_dataset['model_output_scaled'], model_dataset['model_cell_id']) 
	cv_splits = list(cv_splits)
	train_idxs, test_idxs = cv_splits[0]
	
	# perform prediction
	model_dataset['model_input_scaled'] = model_dataset['scaler_model_input'].transform(model_dataset['model_input'])   # scale input

	output_pred_full = None
	output_pred_tfl  = None
	pred_idxs = test_idxs
	output_true = model_dataset['model_output'][pred_idxs]
	
	# predictions - scaled
	output_pred_full_sc = model_full.predict( model_dataset['model_input_scaled'][pred_idxs], verbose=False)
	output_pred_tfl_sc = model_tfl.predict( model_dataset['model_input_scaled'][pred_idxs], verbose=False)
		
	# unscale predictions
	output_pred_full = model_dataset['scaler_model_output'].inverse_transform( output_pred_full_sc )
	output_pred_tfl = model_dataset['scaler_model_output'].inverse_transform( output_pred_tfl_sc )   
	
	# scale units for DCIR features
	for i in range(1,7):
		output_true[:,i] *= 1000
		output_pred_full[:,i] *= 1000
		output_pred_tfl[:,i] *= 1000
   
	# get MAPE and RMSE
	mape_full, rmse_full = get_prediction_error(output_true, output_pred_full)
	mape_tfl, rmse_tfl = get_prediction_error(output_true, output_pred_tfl)
	
	
	# create plots 
	fig = plt.figure(figsize=(8,6), constrained_layout=True)
	gs = GridSpec(2, 2, figure=fig)
	
	# create colormaps
	cmap = plt.cm.get_cmap('Blues')
	vmin = min(output_true[:,0]) * 0.95
	vmax = max(output_true[:,0])
	
	# Set title 
	feature_names = ['Discharge Capacity', 
					 'Charge Resistance at 20% SOC', 'Charge Resistance at 50% SOC', 'Charge Resistance at 90% SOC',
					 'Discharge Resistance at 20% SOC', 'Discharge Resistance at 50% SOC', 'Discharge Resistance at 90% SOC']
	feature_units = ['Ahr', 'm$\Omega$', 'm$\Omega$', 'm$\Omega$', 'm$\Omega$', 'm$\Omega$', 'm$\Omega$', 'm$\Omega$']
	feature_idx = np.argwhere(np.array(all_features) == feature)[0][0]
	feature_str = feature_names[feature_idx]
	feature_units_str = feature_units[feature_idx]
	fig.suptitle("Quantization Comparision for {}".format(feature_str))
	
	# define tick marks --> force all subplots to use same scale
	buffer = 0.025
	share_ticks = True
	all_model_outputs = np.hstack([output_true[:,feature_idx], output_pred_full[:,feature_idx], output_pred_tfl[:,feature_idx]])
	if feature_idx == 0:
		feature_min = np.floor(np.min( all_model_outputs )*100) / 100 * (1-buffer)
		feature_max =  np.ceil(np.max( all_model_outputs )*100) / 100 * (1+buffer)
	else:
		feature_min = np.floor( np.min( all_model_outputs ) ) * (1-buffer)
		feature_max =  np.ceil( np.max( all_model_outputs ) ) * (1+buffer)
	feature_ticks_lg = np.around( np.linspace(feature_min, feature_max, 7, endpoint=True), (2 if feature_idx == 0 else 1))
	feature_ticks_sm = np.around( np.linspace(feature_min, feature_max, 5, endpoint=True), (2 if feature_idx == 0 else 1))
	
	# Panel 1: Full Model Feature Accuracy
	ax1 = fig.add_subplot(gs[0,0])
	ax1.set_title("Full Model Feature Accuracy")
	ax1.set_ylabel("True [{}]".format(feature_units_str))
	ax1.set_xlabel("Predicted [{}]".format(feature_units_str))
	min_val = min(output_true[:,feature_idx])
	max_val = max(output_true[:,feature_idx])
	if share_ticks:
		ax1.set_xlim([feature_min, feature_max])
		ax1.set_ylim([feature_min, feature_max])
		ax1.set_xticks(feature_ticks_sm)
		ax1.set_yticks(feature_ticks_sm)
		min_val = min(feature_ticks_sm)
		max_val = max(feature_ticks_sm)
	ax1.scatter(output_pred_full[:,feature_idx], output_true[:,feature_idx], c=output_true[:,0], cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, edgecolors='k')
	
	ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
	r2 = r2_score(output_true[:,feature_idx], output_pred_full[:,feature_idx])
	ax1.annotate("MAPE: {}%".format(round(mape_full[feature_idx],3)), xy=(0.01,0.90), xycoords='axes fraction' )
	ax1.annotate("RMSE: {}{}".format(round(rmse_full[feature_idx],3),feature_units_str), xy=(0.01,0.80), xycoords='axes fraction' )
	ax1.annotate("$R^2$: {}".format(round(r2,3)), xy=(0.01,0.70), xycoords='axes fraction' )
	
	# Panel 2: Quantized Model Feature Accuracy
	ax2 = fig.add_subplot(gs[0, 1])
	ax2.set_title("Quantized Model Feature Accuracy")
	ax2.set_ylabel("True [{}]".format(feature_units_str))
	ax2.set_xlabel("Predicted [{}]".format(feature_units_str))
	min_val = min(output_true[:,feature_idx])
	max_val = max(output_true[:,feature_idx])
	if share_ticks:
		ax2.set_xlim([feature_min, feature_max])
		ax2.set_ylim([feature_min, feature_max])
		ax2.set_xticks(feature_ticks_sm)
		ax2.set_yticks(feature_ticks_sm)
		min_val = min(feature_ticks_sm)
		max_val = max(feature_ticks_sm)
	
	ax2.scatter(output_pred_tfl[:,feature_idx], output_true[:,feature_idx], c=output_true[:,0], cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, edgecolors='k')
	ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
	r2 = r2_score(output_true[:,feature_idx], output_pred_tfl[:,feature_idx])
	ax2.annotate("MAPE: {}%".format(round(mape_tfl[feature_idx],3)), xy=(0.01,0.90), xycoords='axes fraction' )
	ax2.annotate("RMSE: {}{}".format(round(rmse_tfl[feature_idx],3),feature_units_str), xy=(0.01,0.80), xycoords='axes fraction' )
	ax2.annotate("$R^2$: {}".format(round(r2,3)), xy=(0.01,0.70), xycoords='axes fraction' )
	
	# Panel 3: Correlation Between Full and Quantized Model
	ax3 = fig.add_subplot(gs[1, :])
	ax3.set_title("Correlation between Full and Quantized Model Feature Predictions")
	ax3.set_ylabel("Full Model Prediction [{}]".format(feature_units_str))
	ax3.set_xlabel("Quantized Model Prediction [{}]".format(feature_units_str))
	min_val = min(output_true[:,feature_idx])
	max_val = max(output_true[:,feature_idx])
	if share_ticks:
		ax3.set_xlim([feature_min, feature_max])
		ax3.set_ylim([feature_min, feature_max])
		ax3.set_xticks(feature_ticks_lg)
		ax3.set_yticks(feature_ticks_lg)
		min_val = min(feature_ticks_lg)
		max_val = max(feature_ticks_lg)
	ax3.scatter(output_pred_tfl[:,feature_idx], output_pred_full[:,feature_idx], c=output_true[:,0], cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, edgecolors='k')
	ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
	r2 = r2_score(output_pred_full[:,feature_idx], output_pred_tfl[:,feature_idx])
	mape, rmse = get_prediction_error(output_pred_full, output_pred_tfl)
	ax3.annotate("MAPE: {}%".format(round(mape[feature_idx],3)), xy=(0.01,0.90), xycoords='axes fraction' )
	ax3.annotate("RMSE: {}{}".format(round(rmse[feature_idx],3),feature_units_str), xy=(0.01,0.80), xycoords='axes fraction' )
	ax3.annotate("$R^2$: {}".format(round(r2,3)), xy=(0.01,0.70), xycoords='axes fraction' )

	# Save plot
	if dir_save is not None:
		f_plot = dir_save.joinpath("Quantization Comparison - {}.png".format(feature_str))
		name_idx = 1
		while f_plot.exists():
			f_plot = dir_save.joinpath("Quantization Comparison - {} {}.png".format(feature_str, name_idx))
			name_idx += 1
		plt.savefig(f_plot, dpi=300)
	plt.show()
   

if __name__ == '__main__':
	# model input type / parameters
	pulse_type = 'chg'  # {'chg', 'dchg'}
	pulse_soc = None    # {20, 50, 90, None}. None will use pulses from all SOCs
	input_size = 100    # pulse will be interpolated to this size (default is 100)

	quantization_level = QUANTIZATION_LEVEL.INT8
	
	num_cv_splits = 3	# number of CV splits to perform
	save_plots = True          # whether to save plots as new files
	save_tfl_model = True     # whether to save new tfl file and C++ header

	#region: model saving parameters / paths
	f_saved_models_folder = dir_results.joinpath("saved_models", "slowpulse", pulse_type.lower())
	today_date_str = datetime.today().strftime("%Y-%m-%d")
	f_save_results_path = f_saved_models_folder.joinpath(today_date_str)
	f_save_results_path.parent.mkdir(parents=True, exist_ok=True)
	#endregion

	#region: load saved keras model and data
	f_saved_models_folder_latest = get_latest_models_folder(f_saved_models_folder)
	if f_saved_models_folder_latest is None:
		raise RuntimeError("There are no saved models in the following directory: \'{}\'".format(f_saved_models_folder))
	f_quantized_model = None
	
	# Load newest pulse model & associated scalers
	print("Loading newest pulse model...")
	f_saved_model = sorted( list(f_saved_models_folder_latest.glob("*.h5")) )[-1]
	f_saved_model_input_scaler = sorted( list(f_saved_models_folder_latest.glob("Model_Scaler_Input*")) )[-1]
	f_saved_model_output_scaler = sorted( list(f_saved_models_folder_latest.glob("Model_Scaler_Output*")) )[-1]

	model_full = tf.keras.models.load_model(f_saved_model)
	model_full_input_scaler = pickle.load(open(f_saved_model_input_scaler, 'rb'))
	model_full_output_scaler = pickle.load(open(f_saved_model_output_scaler, 'rb'))
	
	# Load dataset
	model_dataset = get_pulse_model_dataset(pulse_type=pulse_type, pulse_soc=pulse_soc, pulse_len=input_size)
	model_dataset['scaler_model_input'] = model_full_input_scaler
	model_dataset['scaler_model_output'] = model_full_output_scaler
	# Ensure input is scaled with loaded scalers
	model_dataset['model_input_scaled'] = model_dataset['scaler_model_input'].transform(model_dataset['model_input'])
	#endregion

	#region: get or create TFL model 
	# create new TFL model from most recent full-model
	if save_tfl_model:
		# Full 8bit quantization requires a representative dataset function
		def representative_dataset():
			frac_samples_to_use = 1.0
			size_input_data = len(model_dataset['model_input_scaled'])
			samples_idxs = np.random.randint( 0, size_input_data, int(frac_samples_to_use*size_input_data) )
			for i in samples_idxs:
				yield [ model_dataset['model_input_scaled'][i].astype('float32') ]

		# Convert model to TFL and apply selected quantization
		print("Quantizing model...")
		converter = tf.lite.TFLiteConverter.from_keras_model(model_full)
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
		if quantization_level == QUANTIZATION_LEVEL.INT8:
			converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
			converter.inference_input_type = tf.uint8
			converter.inference_output_type = tf.uint8
			converter.representative_dataset = representative_dataset
		elif quantization_level == QUANTIZATION_LEVEL.FLOAT16:
			converter.target_spec.supported_types = [tf.float16]
		model_quantized = converter.convert()

		# Save TFL model
		print("Saving quantized model as C++ header file...")
		filename = "{}_Quantized_{}.tflite".format( str(f_saved_model.name)[:str(f_saved_model.name).rfind('.h5')],
													str(quantization_level.name) )
		f_quantized_model = f_saved_model.parent.joinpath( filename )
		f_quantized_model.write_bytes(model_quantized)

		# Create and save C++ header file
		filename = "{}.h".format( str(f_quantized_model.name)[:str(f_quantized_model.name).rfind('.')] )
		f_quantized_model_header_file = f_quantized_model.parent.joinpath(filename)
		with open(f_quantized_model_header_file, 'w') as file:
			file.write( tfl_to_hex_array(model_full, model_quantized, f_quantized_model, 
										scalers=(model_dataset['scaler_model_input'], model_dataset['scaler_model_output'])))
			file.close()
	# Load previously saved TFL model
	else:
		print("Loading most recent, saved TFL model...")
		f_tfl_models = sorted(list(f_saved_models_folder_latest.glob("*.tflite")))
		if len(f_tfl_models) == 0:
			raise RuntimeError("There are no saved TFL models. Please set \'save_tfl_model\' to \'True\' and re-run this script.")
		f_quantized_model = f_tfl_models[-1]
	#endregion


	#region: perform prediction
	tfl_model = TFL_Model_Wrapper(f_quantized_model)
	model_dataset['model_input_scaler'] = model_full_input_scaler
	model_dataset['model_output_scaler'] = model_full_output_scaler
	pred_results = perform_prediction(tfl_model, model_dataset, num_cv_splits, rand_seed)

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

	# plot quantized model train & test
	print("Plotting quantized model train and test accuracy")
	f_save_plots_path = dir_figures.joinpath("raw", "_slowpulse_py_", f"pulse_type={pulse_type}")
	f_save_plots_path.mkdir(parents=True, exist_ok=True)
	plot_predictions(pred_results, "Slow Pulse (Quantized)", save_fig=True, save_path=f_save_plots_path)

	# plot full v. quantized comparison for each output feature
	all_features = ['dchg_q', 
				'dcir_chg_20soc', 'dcir_chg_50soc', 'dcir_chg_90soc',
				'dcir_dchg_20soc', 'dcir_dchg_50soc', 'dcir_dchg_90soc' ]
	for feature in all_features:
		plot_quantization_comparison(model_full, tfl_model, model_dataset, feature=feature, dir_save=f_save_plots_path)
	#endregion

	print("Quantization.py complete")
	