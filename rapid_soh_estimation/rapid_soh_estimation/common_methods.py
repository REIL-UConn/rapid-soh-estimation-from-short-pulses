
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import *


def get_group_id(cell_id:int) -> int:
    """Obtains the group id corresponding to the given cell id

    Args:
        cell_id (int): cell id to find group for

    Returns:
        int: group id corresponding to the given cell id
    """
    assert cell_id in df_test_tracker['Cell ID'].unique()
    temp = df_test_tracker.loc[df_test_tracker['Cell ID'] == cell_id, 'Group'].values
    assert len(temp) == 1, "Could not find group id for this cell"
    return int(temp[0])

def get_cell_ids_in_group(group_id:int) -> np.ndarray:
	"""Gets all cell ids in the specified group id

	Args:
		group_id (int): The id of the group for which to return the cell ids 

	Returns:
		np.ndarray: An array of all cell_ids in the specified group
	"""
	assert group_id in df_test_tracker['Group'].unique(), f"Invalid group id entered: {group_id}. The group id must be one of the following: {df_test_tracker['Group'].unique()}"
	start = (group_id - 1) * 6
	end = min(start+6, 64)
	return np.arange(start, end, 1) + 2

class Custom_CVSplitter():
	"""A custom cross-validation split wrapper. Allows for splitting by group_id or cell_id and returns n_splits number of cross validation folds
	"""

	def __init__(self, n_splits=3, split_type='group_id', rand_seed=None):
		assert isinstance(n_splits, int), "\'n_splits\' must be an interger value"
		self.n_splits = n_splits
		self.allowed_split_types = ['group_id', 'cell_id']
		assert split_type in self.allowed_split_types, "ValueError. \'split type\' must be one of the following: {}".format(self.allowed_split_types)
		self.split_type = split_type
		self.rand_seed = rand_seed
		
	def get_n_splits(self, X, y, groups):
		return self.n_splits

	def split(self, X, y, cell_ids):
		'given input data (X) and output data (y), returns (train_idxs, test_idxs) --> idxs are relative to X & y'
		kf = None
		if self.rand_seed is None:
			kf = KFold(n_splits=self.n_splits, shuffle=True)
		else:
			kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.rand_seed)
		
		if self.split_type == self.allowed_split_types[0]:      # 'group_id'
			group_ids = np.arange(1, 12, 1)
			# for every cv split (by group), convert group_id_idxs to X & y idxs
			for train_group_idxs, test_group_idxs in kf.split(group_ids):
				train_idxs = []
				test_idxs = []
				train_groups = group_ids[train_group_idxs]
				test_groups = group_ids[test_group_idxs]
				# go through all train group ids in this split
				for train_group_id in train_groups: 
					train_cell_ids = get_cell_ids_in_group(train_group_id)
					# add X & y idxs where cell_id is equal to each cell in this group
					for cell_id in train_cell_ids:
						cell_idxs = np.hstack(np.argwhere( cell_ids == cell_id ))
						train_idxs.append(cell_idxs)
				# go through all test group ids in this split
				for test_group_id in test_groups:
					test_cell_ids = get_cell_ids_in_group(test_group_id)
					# add X & y idxs where cell_id is equal to each cell in this group
					for cell_id in test_cell_ids:
						cell_idxs = np.hstack(np.argwhere( cell_ids == cell_id ))
						test_idxs.append(cell_idxs)

				train_idxs = np.hstack(train_idxs)
				test_idxs = np.hstack(test_idxs)
				yield train_idxs, test_idxs
				
		elif self.split_type == self.allowed_split_types[1]:      # 'cell_id'
			# for every cv split (by cell), convert cell_id_idxs to X & y idxs
			for train_cell_idxs, test_cell_idxs in kf.split(np.unique(cell_ids)):
				train_idxs = []
				test_idxs = []
				train_cells = np.unique(cell_ids)[train_cell_idxs]
				test_cells = np.unique(cell_ids)[test_cell_idxs]
				
				# go through all train group ids in this split
				for train_cell_id in train_cells: 
					cell_idxs = np.hstack(np.argwhere(cell_ids == train_cell_id))
					train_idxs.append(cell_idxs)
					
				# go through all test group ids in this split
				for test_cell_id in test_cells:
					cell_idxs = np.hstack(np.argwhere(cell_ids == test_cell_id))
					test_idxs.append(cell_idxs)
			
				train_idxs = np.hstack(train_idxs)
				test_idxs = np.hstack(test_idxs)
				yield train_idxs, test_idxs

def create_modeling_data(all_data:dict, input_feature_keys:list, output_feature_keys:list=['q_dchg', 'dcir_chg_20', 'dcir_chg_50', 'dcir_chg_90', 'dcir_dchg_20', 'dcir_dchg_50', 'dcir_dchg_90']) -> dict:
	"""Returns new dictionary with 'model_input' and 'model_output' keys for simpler model training

	Args:
		all_data (dict): All data from which to extrac the specified input and output feature
		input_feature_keys (list): A list of keys (that exist in all_data.keys()) to use as model input
		output_feature_keys (list, optional): A list of keys (that exist in all_data.keys()) to use as model output. Defaults to ['q_dchg', 'dcir_chg_20', 'dcir_chg_50', 'dcir_chg_90', 'dcir_dchg_20', 'dcir_dchg_50', 'dcir_dchg_90'].

	Returns:
		dict: A new dict with keys: ['cell_id', 'group_id', 'rpt', 'model_input', 'model_output']
	"""
	assert len(input_feature_keys) > 0
	for f in input_feature_keys: assert f in list(all_data.keys())
	for f in output_feature_keys: assert f in list(all_data.keys())

	modeling_dic = {
		'cell_id':all_data['cell_id'],
		'group_id':all_data['group_id'],
	 	'rpt':all_data['rpt'],
		'model_input':[],
		'model_output':[],
	}
	if len(input_feature_keys) == 1:
		modeling_dic['model_input'] = all_data[input_feature_keys[0]]

	for i in range(len(all_data['cell_id'])):
		if len(input_feature_keys) > 1:
			modeling_dic['model_input'].append( [all_data[f_key][i] for f_key in input_feature_keys] )
		modeling_dic['model_output'].append( [all_data[f_key][i] for f_key in output_feature_keys] )

	modeling_dic['model_input'] = np.asarray(modeling_dic['model_input'])
	modeling_dic['model_output'] = np.asarray(modeling_dic['model_output'])
	return modeling_dic

def interp_time_series(ts:np.ndarray, ys:np.ndarray, n_points:int) -> tuple:
    """Interpolates all y arrays to n_points based on a shared time array

    Args:
        ts (np.ndarray): An array of time values corresponding to every entry in ys
        ys (np.ndarray): A single array (or several stacked arrays) of values corresponding to ts
        n_points (int): The output length of ts and ys

    Returns:
        tuple: A tuple of interpolated time values and corresponding y values (ts_interp, ys_interp). *Note that ys_interp will have the same shape as ys*
    """
    ts_interp = np.linspace(ts[0], ts[-1], n_points)
    ys_interp = None

    if len(ys.shape) == 1:
        # only a single array of y-values was passed
        f = interpolate.PchipInterpolator(ts, ys)
        ys_interp = f(ts_interp)
    else:
        ys_interp = []
        for y in ys:
            f = interpolate.PchipInterpolator(ts, y)
            ys_interp.append( f(ts_interp) )
        ys_interp = np.asarray(ys_interp)

    return ts_interp, ys_interp

def clean_time_series_features(ts:np.ndarray, ys:np.ndarray) -> tuple:
    """Removes duplicate timestamps and corresponding entries in ys

    Args:
        ts (np.ndarray): array of time values corresponding to every entry in ys
        ys (np.ndarray): a single array (or several stacked arrays) of values corresponding to ts

    Returns:
        tuple: (ts_clean, ys_clean). *Note that ys_clean will have the same data type as ys*
    """
    ts_clean, idxs = np.unique(ts, return_index=True)
    ys_clean = None
    if len(ys.shape) == 1:
        ys_clean = ys[idxs]
    else:
        ys_clean = []
        for y in ys:
            ys_clean.append( y[idxs] )
        ys_clean = np.asarray(ys_clean)

    return ts_clean, ys_clean

def get_preprocessed_data_files(dir_preprocessed_data:Path, data_type:str, cell_id:int):
    """Returns a list of Path objects to all pkl files containing data for this cell

    Args:
        dir_preprocessed_data (Path): location of downloaded preprocessed data
        data_type (str): {'rpt', 'cycling'}. Wether to look for RPT or Cycling data
        cell_id (int): The cell id to find data for

    Returns:
        list: list of Path objects
    """
    assert data_type in ['rpt', 'cycling']

    dir_data = dir_preprocessed_data.joinpath(f'{data_type}_data')
    all_files = list(dir_data.glob(f'{data_type}_cell_{cell_id:02d}*'))

    def _file_part_num(file_path:Path):
        file_str = str(file_path.name)
        return int(file_str[file_str.rindex('_part') + len('_part') : file_str.rindex('.pkl')])
    
    if len(all_files) == 0 or '_part' not in str(all_files[0]): 
        return all_files
    else:
        return sorted(all_files, key=_file_part_num)
	
def load_preprocessed_data(file_paths) -> pd.DataFrame:
    """Loads the processed data contained at the provided file path(s). Use 'get_preprocessed_data_files()' to get all file paths

    Args:
        file_paths (Path or list): Path or list of Path objects. 

    Returns:
        pd.DataFrame: A dataframe containing the data at the provided filepath .If multiple file paths are provided, the data will be concatenated into a single dataframe
    """

    if hasattr(file_paths, '__len__'):
        all_data = []
        if len(file_paths) == 0:
            print("WARNING: The provided list of filepaths is empty. Returning None")
            return None
        for file_path in file_paths:
            all_data.append( pickle.load(open(file_path, 'rb')) )
        return pd.concat(all_data, ignore_index=True)
    else:
        return pickle.load(open(file_paths, 'rb'))

def load_processed_data(data_type:str, filename:str=None) -> dict:
	"""Loads saved processed data for the specified data type

	Args:
		data_type (str): {'cc', 'slowpulse', 'fastpulse', 'ultrafastpulse'}. The data_type the data corresponds to.
		filename (str, optional): Can optionally specify the filename of the saved data. If not provided, the most recent auto-named file will be returned.

	Returns:
		dict: The saved data
	"""
	assert data_type in ['cc', 'slowpulse', 'fastpulse', 'ultrafastpulse']

	if filename is not None:
		f = dir_processed_data.joinpath("LFP", data_type, filename)
		if not f.exists(): 
			raise ValueError(f"Could not find specified file: {f}")
		return pickle.load(open(f, 'rb'))
	else:
		prev_files = sorted(dir_processed_data.joinpath("LFP", data_type).glob(f"processed_data_{data_type}_*"))
		if len(prev_files) == 0: 
			raise ValueError("Could not find any previously saved files. Try providing a filename")
		else:
			return pickle.load(open(prev_files[-1], 'rb'))


def create_model(n_hlayers, n_neurons, act_fnc, opt_fnc, learning_rate, input_size=100):
    """Creates a keras sequential model with specified parameters
    n_hlayers: number of hidden layers to use
    n_neurons: number of neurons per hidden layer
    act_fnc: activation function to use (\'tanh\', \'relu\',etc)
    opt_fnc: optimizer function to use (\'sgd\', \'adam\', etc)
	"""
    
    
    # add input layer to Sequential model
    model = keras.models.Sequential()
    model.add( keras.Input(shape=(input_size,)) )
    # model.add( keras.layers.Dense(units=n_neurons, activation=act_fnc, input_dim=input_size) )

    # add hidden layers
    for i in range(n_hlayers):
        model.add( keras.layers.Dense(units=n_neurons, activation=act_fnc) )
        
    # add output layer
    model.add( keras.layers.Dense(7) )
    
    # compile model with chosen metrics
    opt = None
    if opt_fnc == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_fnc == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError("opt_func must be either \'adam\' or \'sgd\'")
    
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.mean_squared_error,      
                  # make sure to normalize all outputs, otherwise DCIR values will drastically skew MSE reading compared to error of predicted SOH
                  metrics=['accuracy'] )
    return model

def plot_predictions(pred_results, model_name, save_fig=False, save_path=None):
    if save_fig:
        assert save_path is not None, "must provide a filepath to save to"
    
    # get min,max of each output feature (for sharing axis limits across figures)
    feature_mins = [99999999999,99999999999,99999999999,99999999999,99999999999,99999999999,99999999999]
    feature_maxs = [-99999999999,-99999999999,-99999999999,-99999999999,-99999999999,-99999999999,-99999999999]
    for cv in range(len(pred_results['cv_splits'])):
        for test_type in ['test','train']:
            for i in range(7):
                feature_mins[i] = min( [min(pred_results[test_type]['y_trues'][cv][:,i]), feature_mins[i]] )
                feature_maxs[i] = max( [max(pred_results[test_type]['y_trues'][cv][:,i]), feature_maxs[i]] )
    
    cv_split_to_plot = 0
    
    # scale DCIR features to mOhm
    for i in range(1,7):
        pred_results['test']['y_preds'][cv_split_to_plot][:,i] *= 1000
        pred_results['test']['y_trues'][cv_split_to_plot][:,i] *= 1000
        pred_results['train']['y_preds'][cv_split_to_plot][:,i] *= 1000
        pred_results['train']['y_trues'][cv_split_to_plot][:,i] *= 1000
        feature_mins[i] *= 1000
        feature_maxs[i] *= 1000 
        
    for test_type in ['train', 'test']:
        output_true = pred_results[test_type]['y_trues'][cv_split_to_plot]
        output_pred = pred_results[test_type]['y_preds'][cv_split_to_plot]
        
        # define tick marks
        buffer = 0.020
        share_ticks = True
        share_dcir_ticks = True
        feature_ticks = []
        for i in range(7):
            if i == 0:
                feature_min = np.floor( feature_mins[0]*100 ) / 100 * (1-buffer)
                feature_max =  np.ceil( feature_maxs[0]*100 ) / 100 * (1+buffer)
                feature_ticks.append( np.around(np.linspace(feature_min, feature_max, 7, endpoint=True), 3) )
            else:
                feature_min = np.floor( feature_mins[i] ) * (1-buffer)
                feature_max =  np.ceil( feature_maxs[i] ) * (1+buffer)
                feature_ticks.append( np.around(np.linspace(feature_min, feature_max, 5, endpoint=True), 1) )
        if share_dcir_ticks: 
            dcir_min = np.floor( min(feature_mins[1:]) ) * (1-buffer)
            dcir_max =  np.ceil( max(feature_maxs[1:]) ) * (1+buffer)
            dcir_ticks = np.around( np.linspace(dcir_min, dcir_max, 5, endpoint=True), 1)
            for i in range(6):
                feature_ticks[i+1] = dcir_ticks

       
        # create plots
        fig = plt.figure(figsize=(10,8), constrained_layout=True)
        gs = GridSpec(3, 3, figure=fig)
        
        # define colormaps
        cmap = plt.cm.get_cmap('Blues')
        vmin = min(output_true[:,0]) * 0.95
        vmax = max(output_true[:,0])
        
        # create plots
        axes = [
            fig.add_subplot(gs[0, :]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]),
            fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])
        ]
        titles = [
            "Discharge Capacity",
            "DCIR (chg, 20% soc)", "DCIR (chg, 50% soc)", "DCIR (chg, 90% soc)",
            "DCIR (dchg, 20% soc)", "DCIR (dchg, 50% soc)", "DCIR (dchg, 90% soc)"
        ]
        xlabels = [
            "Predicted [Ahr]", 
            "Predicted [m$\Omega$]", "Predicted [m$\Omega$]", "Predicted [m$\Omega$]",
            "Predicted [m$\Omega$]", "Predicted [m$\Omega$]", "Predicted [m$\Omega$]"
        ]
        ylabels = [
            "True [Ahr]",
            "True [m$\Omega$]", "True [m$\Omega$]", "True [m$\Omega$]",
            "True [m$\Omega$]", "True [m$\Omega$]", "True [m$\Omega$]"
        ]
        units = [
            "Ahr", 
            "m$\Omega$", "m$\Omega$", "m$\Omega$",
            "m$\Omega$", "m$\Omega$", "m$\Omega$",
        ]
        for i in range(7):
            axes[i].set_title(titles[i])
            axes[i].set_xlabel(xlabels[i])
            axes[i].set_ylabel(ylabels[i])
            min_val = min(output_true[:,i])
            max_val = max(output_true[:,i])
            if share_ticks:
                axes[i].set_xticks(feature_ticks[i])
                axes[i].set_yticks(feature_ticks[i])
                tick_min, tick_max = min(feature_ticks[i]), max(feature_ticks[i])
                axes[i].set_xlim([tick_min, tick_max])
                axes[i].set_ylim([tick_min, tick_max])
                min_val = tick_min
                max_val = tick_max
            
            axes[i].scatter(output_pred[:,i], output_true[:,i], c=output_true[:,0], cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, edgecolors='k')
            axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)    


            # below commented code displays the 3-fold mean errors
            # mape = pred_results[test_type]['MAPE'][0][i]
            # rmse = pred_results[test_type]['RMSE'][0][i]
    
            # below code displays the errors for shown plot
            mape = mean_absolute_percentage_error(output_true[:,i], output_pred[:,i])*100
            rmse = root_mean_squared_error(output_true[:,i], output_pred[:,i])
            r2 = pred_results[test_type]['R2'][cv_split_to_plot][i]
            
            axes[i].annotate("MAPE: {}%".format(round(mape, 3)), 
                            xy=(0.01 if i == 0 else 0.03,0.90), xycoords='axes fraction' )
            axes[i].annotate("RMSE: {}{}".format(round(rmse, 3), units[i]), 
                            xy=(0.01 if i == 0 else 0.03,0.80), xycoords='axes fraction' )
            axes[i].annotate("$R^2$: {}".format(round(r2, 3)), 
                            xy=(0.01 if i == 0 else 0.03,0.70), xycoords='axes fraction' )
        
        fig_name = "{} {} Accuracy".format(model_name, test_type.capitalize())
        fig.suptitle(fig_name)
        if save_fig:
            f_plot = save_path.joinpath("{}.pdf".format(fig_name))
            name_idx = 1
            while f_plot.exists():
                f_plot = save_path.joinpath("{} - {}.pdf".format(fig_name, name_idx))
                name_idx += 1
            plt.savefig(f_plot, dpi=300)
        plt.show()

def perform_prediction(model, model_dataset, num_cv=3, rand_seed=123, split_type='group_id'):
    req_keys = ['model_cell_id', 'model_input', 'model_input_scaler', 'model_output', 'model_output_scaler']
    for key in req_keys:
        assert key in model_dataset.keys()
        
    # scale input & output data
    model_dataset['model_input_sc'] = model_dataset['model_input_scaler'].transform( model_dataset['model_input'] )
    model_dataset['model_output_sc'] = model_dataset['model_output_scaler'].transform( model_dataset['model_output'] )
    
    # perform 3-fold CV splits
    cvSplitter = Custom_CVSplitter(n_splits=num_cv, rand_seed=rand_seed, split_type=split_type)
    cv_splits = cvSplitter.split( model_dataset['model_input_sc'], 
                                  model_dataset['model_output_sc'], 
                                  model_dataset['model_cell_id'] ) 
    cv_splits = list(cv_splits)
    
    # perform prediction
    # run [num_cv_splits]-fold CV and get average prediction accuracy  
    return_dic = {
        'cv_splits':cv_splits,
        'test':{
            'MAPE':[None, None], 'RMSE':[None, None], 
            'R2':[], 'y_preds':[], 'y_trues':[], 'cell_ids':[], 'socs':[]
        },
        'train':{
            'MAPE':[None, None], 'RMSE':[None, None], 
            'R2':[], 'y_preds':[], 'y_trues':[], 'cell_ids':[], 'socs':[]
        }
    }
    results = {}
    cv_idx = 0
    print("Performing {}-fold Cross Validation:".format(num_cv))
    for train_idxs, test_idxs in cv_splits:
        print("  CV fold {}".format(cv_idx))

        # get test accuracy
        y_pred_scaled = model.predict( model_dataset['model_input_sc'][test_idxs], verbose=False)
        y_pred = model_dataset['model_output_scaler'].inverse_transform(y_pred_scaled)
        test_error = get_prediction_error( model_dataset['model_output'][test_idxs], y_pred )

        # get train accuracy
        X_pred_scaled = model.predict( model_dataset['model_input_sc'][train_idxs], verbose=False )
        X_pred = model_dataset['model_output_scaler'].inverse_transform(X_pred_scaled)
        train_error = get_prediction_error( model_dataset['model_output'][train_idxs], X_pred )

        # get R2 scores
        r2_test = []
        r2_train = []
        for i in range(7):
            r2_test.append( r2_score(model_dataset['model_output'][test_idxs][:,i], y_pred[:,i]) )
            r2_train.append( r2_score(model_dataset['model_output'][train_idxs][:,i], X_pred[:,i]) )
        
        # save results to return from func
        return_dic['test']['y_preds'].append(y_pred)
        return_dic['test']['y_trues'].append(model_dataset['model_output'][test_idxs])
        return_dic['test']['R2'].append(r2_test)
        return_dic['test']['cell_ids'].append(model_dataset['model_cell_id'][test_idxs])
        if 'model_soc' in model_dataset.keys():
            return_dic['test']['socs'].append(model_dataset['model_soc'][test_idxs])
        
        return_dic['train']['y_preds'].append(X_pred)
        return_dic['train']['y_trues'].append(model_dataset['model_output'][train_idxs])
        return_dic['train']['R2'].append(r2_train)
        return_dic['train']['cell_ids'].append(model_dataset['model_cell_id'][train_idxs])
        if 'model_soc' in model_dataset.keys():
            return_dic['train']['socs'].append(model_dataset['model_soc'][train_idxs])
        
        # record test and train error in gridsearch_results
        results['CV {}'.format(cv_idx)] = {
            'train_error': train_error,
            'test_error': test_error
        }
        cv_idx += 1

    # get prediction MAPE and RMSE w/ corresponding confidence intervals
    for key in ['test', 'train']:
        RMSE_vals = np.zeros((num_cv,7))
        MAPE_vals = np.zeros((num_cv,7))
        for i, cv_key in enumerate(results.keys()):
            for j in range(7):
                MAPE_vals[i][j] = results[cv_key]['{}_error'.format(key)][0][j]
                RMSE_vals[i][j] = results[cv_key]['{}_error'.format(key)][1][j]
        mean_MAPE = [ np.mean(MAPE_vals[:,0]), 
                    np.mean(MAPE_vals[:,1]), np.mean(MAPE_vals[:,2]), np.mean(MAPE_vals[:,3]),
                    np.mean(MAPE_vals[:,4]), np.mean(MAPE_vals[:,5]), np.mean(MAPE_vals[:,6]) ]
        std_MAPE = [ np.std(MAPE_vals[:,0]), 
                    np.std(MAPE_vals[:,1]), np.std(MAPE_vals[:,2]), np.std(MAPE_vals[:,3]),
                    np.std(MAPE_vals[:,4]), np.std(MAPE_vals[:,5]), np.std(MAPE_vals[:,6]) ]
        mean_RMSE = [ np.mean(RMSE_vals[:,0]), 
                    np.mean(RMSE_vals[:,1]), np.mean(RMSE_vals[:,2]), np.mean(RMSE_vals[:,3]),
                    np.mean(RMSE_vals[:,4]), np.mean(RMSE_vals[:,5]), np.mean(RMSE_vals[:,6]) ]
        std_RMSE = [ np.std(RMSE_vals[:,0]), 
                    np.std(RMSE_vals[:,1]), np.std(RMSE_vals[:,2]), np.std(RMSE_vals[:,3]),
                    np.std(RMSE_vals[:,4]), np.std(RMSE_vals[:,5]), np.std(RMSE_vals[:,6]) ]
        
        return_dic[key]['MAPE'][0] = mean_MAPE
        return_dic[key]['MAPE'][1] = std_MAPE
        return_dic[key]['RMSE'][0] = mean_RMSE
        return_dic[key]['RMSE'][1] = std_RMSE

        # print("Mean MAPE values: {}".format(mean_MAPE))
        # print("STD of MAPE values: {}".format(std_MAPE))
        # print("Mean RMSE values: {}".format(mean_RMSE))
        # print("STD of RMSE values: {}".format(std_RMSE))
    
    return return_dic

def get_prediction_error(y_true, y_predicted):
    '''returns tuple of (MAPE, RMSE) for prrovided true, predicted values'''
    mape = []
    rmse = []
    # print("y_pred size: ", np.size(y_predicted, axis=1) )
    # print("y_true size: ", np.size(y_true, axis=1))
    if len(np.shape(y_true)) > 1:
        for i in range(0, np.size(y_predicted, axis=1)):
            mape.append(np.round(mean_absolute_percentage_error(y_true[:,i], y_predicted[:,i])*100, 4))
            rmse.append(np.round(root_mean_squared_error(y_true[:,i], y_predicted[:,i]), 4))
    else:
        mape.append(np.round(mean_absolute_percentage_error(y_true, y_predicted)*100, 4))
        rmse.append(np.round(root_mean_squared_error(y_true, y_predicted), 4))
    mape = np.vstack(mape)
    rmse = np.vstack(rmse)
    return mape.reshape(-1), rmse.reshape(-1)

def get_gridsearch_run(gridsearch_params):
    """generates unique gridsearch parameters"""
    for param_vals in product(*gridsearch_params.values()):
        yield dict(zip(gridsearch_params.keys(), param_vals))
    
def run_custom_gridsearch(dataset, gridsearch_params, n_cv_splits=3, verbose=False, record_flops=False, split_type='group_id'):
    """Performs a gridsearchCV on provided dataset, returns results in a single dictionary
    dataset must contain the following keys: \'model_input_scaled\', \'model_output_scaled\', \'model_cell_id\', \'model_input_scaler\', \'model_output_scaler\'
    """
    
    # error checking: make sure dataset contains required keys
    req_dataset_keys = ['model_input_scaled', 'model_output_scaled', 'model_cell_id', 'model_input_scaler', 'model_output_scaler']
    for key in req_dataset_keys:
        assert key in dataset.keys(), "KeyError. \'dataset\' must contain the following keys: {}".format(req_dataset_keys)
    
    # define early stop callback for CV training
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, 
                                                verbose=False, mode='auto', baseline=None, 
                                                restore_best_weights=True)
    
    # generate cross validation splits
    cvSplitter = Custom_CVSplitter(n_splits=n_cv_splits, split_type=split_type)
    cv_splits = cvSplitter.split(dataset['model_input_scaled'], dataset['model_output_scaled'], dataset['model_cell_id']) 
    cv_splits = list(cv_splits)
    
    #region: dictionary to store gridsearch results
    # dic
    #   gridsearch_idx=0
    #       'params' = {}
    #       cv=1:
    #           errors
    #       cv=2:
    #           errors
    #   gridsearch_idx=1
    #       'params' = {}
    #       cv=1:
    #           errors
    #       cv=2:
    #           errors
    #endregion
    gridsearch_results = {}
    gridsearch_idx = 0
    # perform cross validation for every possible combination of provided gridsearch parameters
    for gridsearch_param in get_gridsearch_run(gridsearch_params):
        if verbose: print("Gridsearch Run {}:".format(gridsearch_idx))
        
        gridsearch_results[gridsearch_idx] = {}
        gridsearch_results[gridsearch_idx]['params'] = deepcopy(gridsearch_param)
        
        # create model for this set of parameters
        fit_params = {}
        if 'epochs' in gridsearch_param.keys():
            fit_params['epochs'] = gridsearch_param['epochs']
            gridsearch_param.pop('epochs')
        if 'batch_size' in gridsearch_param.keys():
            fit_params['batch_size'] = gridsearch_param['batch_size']
            gridsearch_param.pop('batch_size')
    
        # add efficiency metrics for each parameter set  TODO: add more metrics other than FLOPS
        if record_flops:
            gridsearch_results[gridsearch_idx]['efficiency'] = {}
            
        # evaluate each cross-validation fold & get average MAPE
        cv_idx = 0
        for train_idxs, test_idxs in cv_splits:
            if verbose: print("  CV fold {}".format(cv_idx))
            try:
                model = create_model(**gridsearch_param, input_size=len(dataset['model_input'][0]))
                if record_flops:
                    flops = get_flops(model, batch_size=1) / (10**9)
                    gridsearch_results[gridsearch_idx]['efficiency']['flops (e9)'] = flops
            except:
                raise RuntimeError(gridsearch_param)
            history = model.fit(dataset['model_input_scaled'][train_idxs], 
                                dataset['model_output_scaled'][train_idxs], 
                                **fit_params,
                                validation_split = 0.1, 
                                callbacks=early_stop, 
                                verbose=False )
    
            # get test accuracy
            y_pred_scaled = model.predict( dataset['model_input_scaled'][test_idxs], verbose=False)
            y_pred = dataset['model_output_scaler'].inverse_transform(y_pred_scaled)
            test_error = get_prediction_error( dataset['model_output'][test_idxs], y_pred )
            
            # get train accuracy
            X_pred_scaled = model.predict( dataset['model_input_scaled'][train_idxs], verbose=False )
            X_pred = dataset['model_output_scaler'].inverse_transform(X_pred_scaled)
            train_error = get_prediction_error( dataset['model_output'][train_idxs], X_pred )

            # record test and train error in gridsearch_results
            gridsearch_results[gridsearch_idx]['CV {}'.format(cv_idx)] = {
                'train_error': train_error,
                'test_error': test_error
            }
            cv_idx += 1
            
        gridsearch_idx += 1
        
    return gridsearch_results

def get_best_gridsearch_run(gridsearch_results):
    """returns best params from gridsearch results"""
    best_run = {
        'params':{},
        'train_error':{'MAPE': [], 'RMSE':[]},
        'test_error':{'MAPE': [], 'RMSE':[]},
    }
    for search_idx in gridsearch_results.keys():
        cv_keys = list(gridsearch_results[search_idx].keys())
        cv_keys.remove('params')
        if 'efficiency' in cv_keys: cv_keys.remove('efficiency')
        for cv_key in cv_keys:
            train_errs = gridsearch_results[search_idx][cv_key]['train_error']  # [MAPE], [RMSE]
            test_errs = gridsearch_results[search_idx][cv_key]['test_error']    # [MAPE], [RMSE]
            if len(best_run['test_error']['MAPE']) == 0 or test_errs[0][0] < best_run['test_error']['MAPE'][0]:
                best_run['params'] = gridsearch_results[search_idx]['params']
                if 'efficiency' in cv_keys: best_run['efficiency'] = gridsearch_results[search_idx]['efficiency']
                best_run['train_error']['MAPE'] = train_errs[0]
                best_run['train_error']['RMSE'] = train_errs[1]
                best_run['test_error']['MAPE'] = test_errs[0]
                best_run['test_error']['RMSE'] = test_errs[1]
                
    return best_run

def get_latest_models_folder(f_saved_models_folder, look_for="*.h5"):
    """returns most recent folder in provided parent directory"""
    f_folders = sorted( [f for f in f_saved_models_folder.glob("*") if not f.is_file()] )
    f_folders.reverse()
    
    for f_folder in f_folders:
        if len(list(f_folder.glob(look_for))) != 0:
            return f_folder
        
    return None




if __name__ == '__main__':
    print('common_methods.py')