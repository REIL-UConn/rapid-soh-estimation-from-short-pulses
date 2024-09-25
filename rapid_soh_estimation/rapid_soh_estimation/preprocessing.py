
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from common_methods import *

# This script pre-processes all raw RPT and cycling data
# The data is saved into RPT Data and Cycling Data folders with one .pkl file per cell in each folder

dir_dropbox = Path("/Users/bnowacki/Library/CloudStorage/Dropbox")
assert dir_dropbox.exists()
dir_data_rpt_raw = dir_dropbox.joinpath('Battery Repurposing Data', 'ILCC RPT Data')
dir_data_cycling_raw = dir_dropbox.joinpath('Battery Repurposing Data', 'ILCC Cycling Data')
dir_data_preprocessed = dir_dropbox.joinpath("Datasets to Publish", "ILCC-LFP-aging-dataset")


def get_neware_data_header_keys(rpt_data:pd.DataFrame) -> tuple:
	"""Provided the details sheet (sheet_name=3 in pd.read_excel) of a saved Neware excel file, returns the columns headers and units

	Args:
		rpt_data (pd.DataFrame): details sheet of neware files

	Returns:
		tuple: (v_key, v_modifier, i_key, i_modifier, q_key, q_modifier)
	"""
	v_key = list(rpt_data.keys())[np.argwhere( np.char.find( list(rpt_data.keys()), 'Voltage' ) == 0 )[0][0]]
	v_modifier = 1 if (v_key.rfind('mV') == -1) else (1/1000)
	i_key = list(rpt_data.keys())[np.argwhere( np.char.find( list(rpt_data.keys()), 'Current' ) == 0 )[0][0]]
	i_modifier = 1 if (i_key.rfind('mA') == -1) else (1/1000)
	q_key = list(rpt_data.keys())[np.argwhere( np.char.find( list(rpt_data.keys()), 'Capacity' ) == 0 )[0][0]]
	q_modifier = 1 if (q_key.rfind('mAh') == -1) else (1/1000)

	return v_key, v_modifier, i_key, i_modifier, q_key, q_modifier

def get_rpt_mapping() -> pd.DataFrame:
	"""
	Returns:
		pd.DataFrame: Returns a df containing mapping of step number to segment key and SOC key, and gives a description of that step
	"""

	df_mapping = pd.DataFrame(columns=['Step', 'Segment Key', 'Pulse Type', 'Pulse SOC'], index=np.arange(0,61.5,1))

	df_mapping['Step'] = np.arange(1,62.5,1).astype(int)
	df_mapping['Segment Key'] = '-'     # {'ref_chg', 'ref_dchg', 'slowpulse', 'fastpulse', 'ultrafastpulse'}
	df_mapping['Pulse Type'] = '-'      # {'chg', 'dchg'}
	df_mapping['Pulse SOC'] = np.nan    # {20, 50, 90}

	# steps for reference discharge and charge segments
	df_mapping.loc[df_mapping['Step'].isin(np.asarray([2,10,18,26, 30,38, 50,54])), 'Segment Key'] = 'ref_chg'
	df_mapping.loc[df_mapping['Step'] == 28, 'Segment Key'] = 'ref_dchg'

	# steps for slow pulses
	for i, soc in enumerate([20,50,90]):
		df_mapping.loc[df_mapping['Step'].isin(np.arange(4+(8*i), 9.5+(8*i), 1)), 'Segment Key'] = 'slowpulse'
		df_mapping.loc[df_mapping['Step'].isin(np.arange(4+(8*i), 6.5+(8*i), 1)), 'Pulse Type'] = 'chg'
		df_mapping.loc[df_mapping['Step'].isin(np.arange(7+(8*i), 9.5+(8*i), 1)), 'Pulse Type'] = 'dchg'
		df_mapping.loc[df_mapping['Step'].isin(np.arange(4+(8*i), 9.5+(8*i), 1)), 'Pulse SOC'] = soc

	# steps for fast pulses
	df_mapping.loc[df_mapping['Step'].isin(np.arange(32, 37.5, 1)), 'Segment Key'] = 'fastpulse'
	df_mapping.loc[df_mapping['Step'].isin(np.arange(32, 37.5, 1)), 'Pulse Type'] = 'chg'
	df_mapping.loc[df_mapping['Step'].isin(np.arange(32, 37.5, 1)), 'Pulse SOC'] = 90
	df_mapping.loc[df_mapping['Step'].isin(np.arange(42, 47.5, 1)), 'Segment Key'] = 'fastpulse'
	df_mapping.loc[df_mapping['Step'].isin(np.arange(42, 47.5, 1)), 'Pulse Type'] = 'dchg'
	df_mapping.loc[df_mapping['Step'].isin(np.arange(42, 47.5, 1)), 'Pulse SOC'] = 90

	# steps for ultra fast pulses
	df_mapping.loc[df_mapping['Step'].isin(np.arange(51, 53.5, 1)), 'Segment Key'] = 'ultrafastpulse'
	df_mapping.loc[df_mapping['Step'].isin(np.arange(51, 53.5, 1)), 'Pulse Type'] = 'chg'
	df_mapping.loc[df_mapping['Step'].isin(np.arange(51, 53.5, 1)), 'Pulse SOC'] = 90
	df_mapping.loc[df_mapping['Step'].isin(np.arange(57, 59.5, 1)), 'Segment Key'] = 'ultrafastpulse'
	df_mapping.loc[df_mapping['Step'].isin(np.arange(57, 59.5, 1)), 'Pulse Type'] = 'dchg'
	df_mapping.loc[df_mapping['Step'].isin(np.arange(57, 59.5, 1)), 'Pulse SOC'] = 90

	return df_mapping

def get_week_num_from_folder_filepath(f_folder:Path) -> float:
	"""Gets the week number from the folder name
	Args:
		f_folder (Path): Path object to location of RPT folder (ex: ../Week 0.0 RPT)
	Returns:
		float: week number as float
	"""

	try:
		filename_str = str(f_folder.name)
		if 'RPT' in filename_str:
			temp = float( filename_str[5:filename_str.rindex('R')-1] )
		elif 'Cycling' in filename_str:
			temp = float( filename_str[5:filename_str.rindex('C')-1] )
		return temp
	except:
		raise ValueError(f"Filename error. Could not find the week number in the following folder: {f_folder.name}. Ensure folder follows the naming convention of \'Week ##.# RPT\'")


def get_channel_from_filename(f_file:Path) -> str:
	"""Gets the channel id from the Neware filename
	Args:
		f_file (Path): Path object to exported Neware data excel file
	Returns:
		str: channel id (ex. '1-1')
	"""
	try:
		splits = str(f_file.name).split('-')
		channel_id = splits[1] + '-' + splits[2]
	except: 
		print(f"Failed to get channel from: {f_file}")
		raise RuntimeError
	assert channel_id in df_test_tracker['Channel'].unique()
	return channel_id

def get_rpt_df_cols():
	"""Predefined set of column names to use when processing RPT data"""
	columns = [
		'Week Number', 'RPT Number', 'Date (yyyy.mm.dd hh.mm.ss)', 
		'Step Number', 'State', 'Time (s)', 'Voltage (V)', 'Current (A)', 'Capacity (Ah)',
		'Segment Key', 'Pulse Type', 'Pulse SOC',
	]
	return columns

def get_cycling_df_cols():
	"""Predefined set of column names to use when processing Cycling data"""
	columns = [
		'Week Number', 'Life', 'Date (yyyy.mm.dd hh.mm.ss)', 'Cycle Number',
		'State', 'Time (s)', 'Voltage (V)', 'Current (A)', 'Capacity (Ah)',
	]
	return columns

def convert_rpt_rel_time_to_float(ts:np.ndarray) -> np.ndarray:
	"""Converts the RPT data under the \'Relative Time(h:min:s.ms\' column to float values

	Args:
		ts (np.ndarray): The values of the \'Relative Time(h:min:s.ms\' column

	Returns:
		np.ndarray: array of continuous time values (in seconds)
	"""
	''''''

	# two steps: convert all time-strings to seconds, then convert relative time to continuous time
	t_seconds = np.zeros_like(ts, dtype=float)
	t_deltas = np.zeros_like(ts, dtype=float)
	t_continuous = np.zeros_like(ts, dtype=float)

	for i, time in enumerate(ts):
		temp = sum(scaler * float(t) for scaler, t in zip([1, 60, 3600], reversed(time.split(":"))))
		t_seconds[i] = temp
		
		if i == 0:
			t_deltas[i] = temp
		else:
			if temp < t_seconds[i-1]:
				t_deltas[i] = temp
			else:
				t_deltas[i] = temp - t_seconds[i-1]
	for i, _ in enumerate(t_deltas):
		if i == 0:
			t_continuous[i] = t_deltas[0]
		else:
			t_continuous[i] = t_continuous[i-1] + t_deltas[i]

	return t_continuous


def extract_data_from_rpt(path_rpt:Path, week_num=None, rpt_num=None) -> pd.DataFrame:
	"""Extracts all useful data from the raw Neware RPT data

	Args:
		path_rpt (Path): Path object to Neware RPT data (excel file)
		week_num (float, optional): If provided, the week number is added to the created dataframe. Defaults to None.
		rpt_num (int, optional): If provided, the rpt number is added to the created dataframe. Defaults to None.

	Returns:
		pd.DataFrame: Processed RPT data in dataframe form
	"""
	assert path_rpt.exists()
	
	# open rpt file
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=UserWarning, module=re.escape('openpyxl.styles.stylesheet'))
		all_rpt_data = pd.read_excel(path_rpt, sheet_name=[0,2,3], engine='openpyxl')
	rpt_data_test_info = all_rpt_data[0]
	rpt_data_stats = all_rpt_data[2]
	rpt_data_details = all_rpt_data[3]
	
	v_key, v_modifier, i_key, i_modifier, q_key, q_modifier = get_neware_data_header_keys(rpt_data_details)
	rpt_mapping = get_rpt_mapping()

	# extract rpt data and store in new dataframe
	new_df = pd.DataFrame(columns=get_rpt_df_cols())
	if week_num is not None: 
		new_df['Week Number'] = np.full(len(rpt_data_details), week_num)
	if rpt_num is not None:
		new_df['RPT Number'] = np.full(len(rpt_data_details), rpt_num)
	new_df['Date (yyyy.mm.dd hh.mm.ss)'] = pd.to_datetime(rpt_data_details['Date(h:min:s.ms)'].values, format="%Y-%m-%d %H:%M:%S")
	new_df['Step Number'] = rpt_data_details['Steps']
	new_df['State'] = rpt_data_details['State']
	new_df['Time (s)'] = convert_rpt_rel_time_to_float(rpt_data_details['Relative Time(h:min:s.ms)'])
	new_df['Voltage (V)'] = rpt_data_details[v_key].astype(float).values * v_modifier
	new_df['Current (A)'] = rpt_data_details[i_key].astype(float).values * i_modifier
	new_df['Capacity (Ah)'] = rpt_data_details[q_key].astype(float).values * q_modifier

	# set 'Segment Key', 'Pulse Type', and 'Pulse SOC' columns
	for col_key in [c for c in rpt_mapping.columns if not c == 'Step']:
		for p_key in rpt_mapping[col_key].unique():
			orig_steps = rpt_mapping.loc[rpt_mapping[col_key] == p_key, 'Step'].values
			true_steps = rpt_data_stats.loc[rpt_data_stats['Original step'].isin(orig_steps), 'Steps']
			idxs = rpt_data_details.loc[rpt_data_details['Steps'].isin(true_steps)].index
			new_df.loc[idxs, col_key] = p_key

	return new_df

def extract_data_from_cycling(path_cycling:Path, week_num=None) -> pd.DataFrame:
	"""Extracts all useful data from the raw Neware cycling data

	Args:
		path_cycling (Path): Path object to Neware cycling data (excel file)
		week_num (float, optional): If provided, the week number is added to the created dataframe. Defaults to None.

	Returns:
		pd.DataFrame: Processed cycling data in dataframe form
	"""
	assert path_cycling.exists()

	# open rpt file
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=UserWarning, module=re.escape('openpyxl.styles.stylesheet'))
		all_cycling = pd.read_excel(path_cycling, sheet_name=[0,2,3], engine='openpyxl')
	cycling_data_test_info = all_cycling[0]
	cycling_data_stats = all_cycling[2]
	cycling_data_details = all_cycling[3]

	protocol_name = cycling_data_test_info.loc[cycling_data_test_info['Test information'] == 'Process name:', 'Unnamed: 1'].values[0]
	is_second_life = 'Second Life' in protocol_name
	
	v_key, v_modifier, i_key, i_modifier, q_key, q_modifier = get_neware_data_header_keys(cycling_data_details)

	new_df = pd.DataFrame(columns=get_cycling_df_cols())
	if week_num is not None: 
		new_df['Week Number'] = np.full(len(cycling_data_details), week_num)
	new_df['Life'] = np.full(len(cycling_data_details), '2nd' if is_second_life else '1st')
	new_df['Date (yyyy.mm.dd hh.mm.ss)'] = pd.to_datetime(cycling_data_details['Date(h:min:s.ms)'].values, format="%Y-%m-%d %H:%M:%S")
	new_df['Cycle Number'] = cycling_data_details['Cycle']
	new_df['State'] = cycling_data_details['State']
	new_df['Time (s)'] = convert_rpt_rel_time_to_float(cycling_data_details['Relative Time(h:min:s.ms)'])
	new_df['Voltage (V)'] = cycling_data_details[v_key].astype(float).values * v_modifier
	new_df['Current (A)'] = cycling_data_details[i_key].astype(float).values * i_modifier
	new_df['Capacity (Ah)'] = cycling_data_details[q_key].astype(float).values * q_modifier

	return new_df


def process_rpt_data(dir_preprocessed_data:Path, file_size_limit_gb=0.100):
	"""Automatically processes all RPT data. Skips files that have already been processed.

	Args:
		dir_preprocessed_data (Path): location of downloaded preprocessed data
		file_size_limit_gb (float, optional): Can optionally set a maximum filesize in gigabytes. Processed data will be split into multiple files if not None. Note that this is not a strict limit as all data from a single week number is saved at once. Defaults to 0.5.
	"""
 
	# get last processed rpt number for each cell
	print("Retreiving last processed RPT for each cell ... this may take a minute")
	last_rpt_cell_map = {c:-1 for c in df_test_tracker['Cell ID'].unique()}
	for c in last_rpt_cell_map.keys():
		file = get_preprocessed_data_files(dir_preprocessed_data, data_type='rpt', cell_id=c)
		if len(file) == 0: 
			last_rpt_cell_map[c] = -1
		else: 
			file = file[-1]
			last_rpt_cell_map[c] = pickle.load(open(file, 'rb'))['RPT Number'].max()
	
	# process all rpt file sequentially (skip already processed ones)
	all_folders = [f for f in dir_data_rpt_raw.glob('*') if f.is_dir()]
	for dir_week in sorted(all_folders, key=get_week_num_from_folder_filepath):
		if not dir_week.is_dir(): continue
		week_num = get_week_num_from_folder_filepath(dir_week)
		rpt_num = int(week_num * 2)
		print(f"Processing Week {week_num}...")

		all_files = [f for f in dir_week.glob('*.xlsx') if f.is_file() and (not str(f.name) == '.DS_Store')]
		for file_rpt in sorted(all_files, key=get_channel_from_filename):
			cell_id = df_test_tracker.loc[df_test_tracker['Channel'] == get_channel_from_filename(file_rpt), 'Cell ID'].values
			assert len(cell_id) == 1
			cell_id = int(cell_id[0])
   
			# skip if already processed
			if rpt_num <= last_rpt_cell_map[cell_id]: continue

			#region: get file name for saving processed data
			filename = None
			filename_idx = None
			if file_size_limit_gb is None:
				filename = dir_preprocessed_data.joinpath("rpt_data", f"rpt_cell_{cell_id:02d}.pkl")
			# create new file if previous file size is too large
			else: 
				# find the next filename_idx of a file with size < file_size_limit_gb
				filename_idx = 0
				filename = dir_preprocessed_data.joinpath("rpt_data", f"rpt_cell_{cell_id:02d}_part{filename_idx:d}.pkl")
				while filename.exists() and ((filename.stat().st_size / 1000000000) >= file_size_limit_gb):
					filename_idx += 1
					filename = dir_preprocessed_data.joinpath("rpt_data", f"rpt_cell_{cell_id:02d}_part{filename_idx:d}.pkl")
			filename.parent.mkdir(parents=True, exist_ok=True)
			#endregion

			#region: create empty dataframe or load previous data for current cell
			df_cell = None
			if filename.exists():
				df_cell = pickle.load(open(filename, 'rb'))
			else:
				df_cell = pd.DataFrame(columns=get_rpt_df_cols())
				# make sure this is Week 00.0 since processed file doesn't exist yet (only for no file limits)
				if file_size_limit_gb is None: assert rpt_num == 0
			#endregion

			#region: process new rpt and concatenate to df_cell
			print(f"  Cell {cell_id} ... ", end='')
			new_df = extract_data_from_rpt(file_rpt, week_num=week_num, rpt_num=rpt_num)
			if df_cell.empty:
				df_cell = new_df
			else:
				df_cell = pd.concat([df_cell, new_df], ignore_index=True)
			#endregion

			pickle.dump(df_cell, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
			print('updated')
	
def process_cycling_data(dir_preprocessed_data:Path, file_size_limit_gb=0.100):
	"""Automatically processes all cycling data. Skips files that have already been processed.

	Args:
		dir_preprocessed_data (Path): location of downloaded preprocessed data
		file_size_limit_gb (float, optional): Can optionally set a maximum filesize in gigabytes. Processed data will be split into multiple files if not None. Note that this is not a strict limit as all data from a single week number is saved at once. Defaults to 0.5.
	"""

	# get last processed week number for each cell
	print("Retreiving last processed cycling data for each cell ... this may take a minute")
	last_week_cell_map = {c:-1 for c in df_test_tracker['Cell ID'].unique()}
	for c in last_week_cell_map.keys():
		file = get_preprocessed_data_files(dir_preprocessed_data, data_type='cycling', cell_id=c)
		if len(file) == 0: 
			last_week_cell_map[c] = -1
		else: 
			file = file[-1]
			last_week_cell_map[c] = pickle.load(open(file, 'rb'))['Week Number'].max()
	
	# process all cycling files sequentially (skip already processed ones)
	all_folders = [f for f in dir_data_cycling_raw.glob('*') if f.is_dir()]
	for dir_week in sorted(all_folders, key=get_week_num_from_folder_filepath):
		if not dir_week.is_dir(): continue
		week_num = get_week_num_from_folder_filepath(dir_week)
		print(f"Processing Week {week_num}...")

		all_files = [f for f in dir_week.glob('*') if f.is_file() and (not str(f.name) == '.DS_Store')]
		for file_rpt in sorted(all_files, key=get_channel_from_filename):
			cell_id = df_test_tracker.loc[df_test_tracker['Channel'] == get_channel_from_filename(file_rpt), 'Cell ID'].values
			assert len(cell_id) == 1
			cell_id = int(cell_id[0])

			# skip if already processed
			if week_num <= last_week_cell_map[cell_id]: continue

			#region: get file name for saving processed data
			filename = None
			filename_idx = None
			if file_size_limit_gb is None:
				filename = dir_preprocessed_data.joinpath("cycling_data", f"cycling_cell_{cell_id:02d}.pkl")
			# create new file if previous file size is too large
			else:
				# find the next filename_idx of a file with size < file_size_limit_gb
				filename_idx = 0
				filename = dir_preprocessed_data.joinpath("cycling_data", f"cycling_cell_{cell_id:02d}_part{filename_idx:d}.pkl")
				while filename.exists() and ((filename.stat().st_size / 1000000000) >= file_size_limit_gb):
					filename_idx += 1
					filename = dir_preprocessed_data.joinpath("cycling_data", f"cycling_cell_{cell_id:02d}_part{filename_idx:d}.pkl")
			filename.parent.mkdir(parents=True, exist_ok=True)
			#endregion

			#region: create empty dataframe or load previous data for current cell
			df_cell = None
			if filename.exists():
				df_cell = pickle.load(open(filename, 'rb'))
			else:
				df_cell = pd.DataFrame(columns=get_cycling_df_cols())
				# make sure this is Week 00.0 since processed file doesn't exist yet (only for no file limits)
				if file_size_limit_gb is None: assert week_num == 0
			#endregion

			#region: process new rpt and concatenate to df_cell
			print(f"  Cell {cell_id} ... ", end='')
			new_df = extract_data_from_cycling(file_rpt, week_num=week_num)
			if df_cell.empty:
				df_cell = new_df
			else:
				df_cell = pd.concat([df_cell, new_df], ignore_index=True)
			#endregion

			pickle.dump(df_cell, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
			print('updated')

def add_life_info_to_rpt_data(dir_preprocessed_data:Path):
	"""Updates RPT dataframes to include a 'Life' column indicating which rows correspond to first life or second life
	Args:
		dir_preprocessed_data (Path): location of downloaded preprocessed data
	"""

	print(f"Updating RPT data with \'Life\' and \'Num Cycles\' information...")
	for cell_id in df_test_tracker['Cell ID'].unique():
		# load all cycling data for this cell
		df_cycling_data = load_preprocessed_data(get_preprocessed_data_files(dir_preprocessed_data, data_type='cycling', cell_id=cell_id))

		# need to iterate through each RPT file separately so we can update it
		all_rpt_files = get_preprocessed_data_files(dir_preprocessed_data, data_type='rpt', cell_id=cell_id)
		for rpt_file in all_rpt_files:
			# load rpt data for this single file
			df_rpt_data = load_preprocessed_data(rpt_file)

			# skip if already added info to this RPT
			if ('Life' in df_rpt_data.columns) and ('Num Cycles' in df_rpt_data.columns): continue

			# get all week numbers in this file and filter cycling data to only this range
			df_cycling_data_filt = df_cycling_data.loc[df_cycling_data['Week Number'] <= df_rpt_data['Week Number'].max()]
			
			# create 'Life' and 'Num Cycles' columns in RPT data
			df_rpt_data['Life'] = np.full(len(df_rpt_data), '1st')
			df_rpt_data['Num Cycles'] = np.full(len(df_rpt_data), 0)

			# set 2nd life based on life info from cycling dataframe
			start_2nd_life = df_cycling_data_filt.loc[df_cycling_data_filt['Life'] == '2nd', 'Week Number'].values
			if len(start_2nd_life) >= 1: 
				start_2nd_life = start_2nd_life[0]
				df_rpt_data.loc[df_rpt_data['Week Number'] > start_2nd_life, 'Life'] = '2nd'

			# set number of cycles based on cycling dataframe
			for rpt_week_num in sorted(df_rpt_data['Week Number'].unique()):
				if rpt_week_num == 0.0: continue
				num_cycles = df_cycling_data_filt.loc[df_cycling_data_filt['Week Number'] == rpt_week_num-0.5, 'Cycle Number'].max()
				df_rpt_data.loc[df_rpt_data['Week Number'] == rpt_week_num, 'Num Cycles'] = num_cycles

			# save updated rpt data file
			pickle.dump(df_rpt_data, open(rpt_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

		print(f"Cell {cell_id} updated with life info and cycle numbers")


if __name__ == '__main__':
	# if using external SSD
	temp = Path("/Volumes/T7/Datasets to Publish/ILCC-LFP-aging-dataset")
	# otherwise can just use dir_data_preprocessed (defined at top of this file)

	process_cycling_data(dir_preprocessed_data=temp)
	process_rpt_data(dir_preprocessed_data=temp)
	add_life_info_to_rpt_data(dir_preprocessed_data=temp)

	print('preprocessing.py complete.\n')