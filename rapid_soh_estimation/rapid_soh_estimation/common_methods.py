
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

def load_processed_data(data_type:str, filepath:str=None) -> dict:
	"""Loads saved processed data for the specified data type

	Args:
		data_type (str): {'cc', 'slowpulse', 'fastpulse', 'ultrafastpulse'}. The data_type the data corresponds to.
		filepath (str, optional): Can optionally specify the file path of the saved data. If not provided, the most recent auto-named file will be returned.

	Returns:
		dict: The saved data
	"""
	assert data_type in ['cc', 'slowpulse', 'fastpulse', 'ultrafastpulse']

	if filepath is not None:
		f = dir_processed_data.joinpath(data_type, filepath)
		if not f.exists(): 
			raise ValueError(f"Could not find specified file: {f}")
		return pickle.load(open(f, 'rb'))
	else:
		prev_files = sorted(dir_processed_data.joinpath(data_type).glob(f"processed_data_{data_type}_*"))
		if len(prev_files) == 0: 
			raise ValueError("Could not find any previously saved files. Try providing a filename")
		else:
			return pickle.load(open(prev_files[-1], 'rb'))



if __name__ == '__main__':
    print('common_methods.py')