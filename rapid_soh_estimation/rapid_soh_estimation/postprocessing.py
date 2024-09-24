
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from common_methods import *

# This script creates a modeling data set from the preprocessed data
# The data is saved under "processed_data"


def get_health_features_from_rpt_data(rpt_data:pd.DataFrame) -> pd.DataFrame:
	"""Extracts the health features (remaining discharge capacity and internal resistance)

	Args:
		rpt_data (pd.DataFrame): The RPT data for a given cell. Use 'load_preprocessed_data()' to get the RPT data.

	Returns:
		pd.DataFrame: A dataframe mapping each RPT to [q_dchg, dcir_chg_20, dcir_chg_50, dcir_chg_90, dcir_dchg_20, dcir_dchg_50, dcir_dchg_90].
	"""
	data_dic = {
		'rpt':[],
		'q_dchg':[],
		'dcir_chg_20':[],
		'dcir_chg_50':[],
		'dcir_chg_90':[],
		'dcir_dchg_20':[],
		'dcir_dchg_50':[],
		'dcir_dchg_90':[],
	}
	for rpt in rpt_data['RPT Number'].unique():
		data_dic['rpt'].append(rpt)
		df_filt = rpt_data.loc[rpt_data['RPT Number'] == rpt]
		# add discharge capacity for the current RPT
		data_dic['q_dchg'].append( df_filt.loc[df_filt['Segment Key'] == 'ref_dchg', 'Capacity (Ah)'].values[-1] )

		# add the six DCIR values for this RPT
		for p_type in ['chg', 'dchg']:
			for soc in [20,50,90]:
				df_pulse = df_filt.loc[(df_filt['Segment Key'] == 'slowpulse') & \
									   (df_filt['Pulse Type'] == p_type) & \
									   (df_filt['Pulse SOC'] == soc)]
				#region: interpolate pulse voltage to exactly 100 seconds 
				# each segment of pulse should be the following lengths (in seconds)
				seg_lengths = [30,10,60]
				seg_ts = []
				seg_vs = []
				for i, step in enumerate(df_pulse['Step Number'].unique()):
					df_seg = df_pulse.loc[df_pulse['Step Number'] == step]
					t = df_seg['Time (s)'].values - df_seg['Time (s)'].values[0]
					t_interp = np.arange(0, seg_lengths[i], 1)
					f_v = interpolate.PchipInterpolator(t, df_seg['Voltage (V)'].values)
					v_interp = f_v(t_interp)
					seg_ts.append(t_interp)
					seg_vs.append(v_interp)
				pulse_v = np.hstack(seg_vs)
				assert len(pulse_v) == 100
				#endregion

				# calculate dcir for this pulse type and soc
				dcir = abs(float((pulse_v[39] - pulse_v[29]) / (1.2 - 0.24)))
				# add this dcir at all three locations (three pulses per rpt)
				data_dic[f'dcir_{p_type}_{soc}'].append(dcir)

	return pd.DataFrame(data_dic)

def extract_cccv_charge(rpt_data:pd.DataFrame, plot_interpolation:bool=False) -> pd.DataFrame:
    """Extracts the CC-CV reference charge data from the preprocessed RPT data

    Args:
        rpt_data (pd.DataFrame): A dataframe containing RPT data. Use 'common_methods.load_processed_data()' to get this information
        plot_interpolation (bool): Whether to plot the interpolation process; only one CC-CV charge cycle will be plotted. See the note below for a better description. Defaults to False.
    Returns:
        pd.DataFrame: A dataframe of the CC-CV charge information (time, voltage, current, capacity) and corresponding week number.
    Notes:
        The reference charge portion of the RPT protocol is interrupted at ~90% SOC to perform a fastpulse. Therefore, to get a continuous voltage profile, the signal must be interpolated over a roughly 4 minute gap in the data. To see the full extent of the interpolation and missing data, set plot_interpolation to True.
    """
    
    # a dataframe to store the interpolated CCCV charge information
    cccv_charge = pd.DataFrame(columns=['RPT Number', 'Time (s)', 'Voltage (V)', 'Current (A)', 'Capacity (Ah)'])

    # extract the CCCV charge data from each RPT
    for i, rpt_num in enumerate(sorted(rpt_data['RPT Number'].unique())):

        # cc charge before pulse interruption
        df_chg_p1 = rpt_data.loc[(rpt_data['RPT Number'] == rpt_num) & \
                                 (rpt_data['Segment Key'] == 'ref_chg') & \
                                 (rpt_data['Step Number'] == 50)]
        # cccv charge after pulse interruption
        df_chg_p2 = rpt_data.loc[(rpt_data['RPT Number'] == rpt_num) & \
                                    (rpt_data['Segment Key'] == 'ref_chg') & \
                                    (rpt_data['Step Number'] == 54)].copy()
        df_chg_p2['Capacity (Ah)'] = df_chg_p2['Capacity (Ah)'].values + df_chg_p1['Capacity (Ah)'].values[-1]
        
        # create dic for current rpt to append to df
        temp_data = {c:None for c in cccv_charge.columns}

        # for each feature, combine both segments and interpolate over entire range (fill in missing data)
        p1_t = df_chg_p1['Time (s)'].values - df_chg_p1['Time (s)'].values[0]
        p2_t = (df_chg_p2['Time (s)'].values - df_chg_p1['Time (s)'].values[0])[90:]
        t_interp = np.arange(p1_t[0], p2_t[-1], 1)
        temp_data['Time (s)'] = t_interp
        for f in ['Voltage (V)', 'Current (A)', 'Capacity (Ah)']:
            f_p1 = df_chg_p1[f].values
            f_p2 = df_chg_p2[f].values[90:]
            f_interp = np.interp(t_interp, np.hstack([p1_t, p2_t]), np.hstack([f_p1, f_p2]))
            # set feature values into temp_data
            temp_data[f] = f_interp
        # set RPT number
        temp_data['RPT Number'] = np.full(len(temp_data['Time (s)']), rpt_num)

        # add data to cccv_charge dataframe
        if cccv_charge.empty:
            cccv_charge = pd.DataFrame(temp_data)
        else:
            cccv_charge = pd.concat([cccv_charge, pd.DataFrame(temp_data)], ignore_index=True)

        # if specified, plot first cycle of interpolation process
        if plot_interpolation and i == 0:
            plt.figure(figsize=(6,2.5))
            plt.plot((df_chg_p1['Time (s)'].values - df_chg_p1['Time (s)'].values[0])/60, 
                     df_chg_p1['Voltage (V)'], 'k.', label='Raw Signal')
            plt.plot((df_chg_p2['Time (s)'].values[90:] - df_chg_p1['Time (s)'].values[0])/60, 
                     df_chg_p2['Voltage (V)'].values[90:], 'k.')
            plt.plot(temp_data['Time (s)']/60, temp_data['Voltage (V)'], 'r-', label='Interpolated')
            plt.xlabel("Time [min]")
            plt.ylabel("Voltage [V]")
            plt.legend(loc='lower right')
            plt.show()

    return cccv_charge

def get_cc_subsamples(voltage_arr:np.ndarray, segment_length:int=600, segment_overlap:float=0.5) -> np.ndarray:
    """Creates a set of subsamples from the full CC charge time-series voltage

    Args:
        voltage_arr (np.ndarray): The voltages during a CC charge sampled at 1Hz
        segment_length (int, optional): The length of each subsample in seconds. Defaults to 600.
        segment_overlap (float, optional): The allowable overlap of subsamples. Must be between 0 and 1 ( a value of 0.0 ensures no overlap). Defaults to 0.5.

    Returns:
        np.ndarray: an array of all subsamples
    """
    assert segment_overlap >= 0 and segment_overlap < 1.0
    
    # the start idx of the current segment
    segment_start_idx = 0
    # the end idx of the current segment
    segment_end_idx = segment_length if len(voltage_arr) > segment_length else len(voltage_arr)-1
    cc_subsamples = []
    while segment_end_idx < len(voltage_arr):
        # add subsample of voltage_arr 
        cc_subsamples.append( voltage_arr[segment_start_idx:segment_end_idx] )
        # if len of current segment < specified length, end loop (at end of voltages)
        if segment_end_idx - segment_start_idx < segment_length: break
        # set new start and end indices
        segment_start_idx = segment_end_idx - int(segment_length*segment_overlap)
        segment_end_idx = (segment_start_idx + segment_length) if (segment_start_idx + segment_length) < len(voltage_arr) else len(voltage_arr)-1

    return cc_subsamples


def create_cc_modeling_data(dir_preprocessed_data:Path, segment_length:int=600, segment_overlap:float=0.5, soc_bounds=(0.3,0.9)) -> dict:
	"""Obtains all 1st-life information used for SOH estimation modeling using segments of the CC charge.

	Args:
		dir_preprocessed_data (Path): location of downloaded preprocessed data
		segment_length (int, optional): The length of each subsample of the full CC-CV charge in seconds. Defaults to 600.
        segment_overlap (float, optional): The allowable overlap of subsamples. Must be between 0 and 1 ( a value of 0.0 ensures no overlap). Defaults to 0.5.
		soc_bounds (tuple, optional): The lower and upper SOC limits (in decimal form) from which to extract the subsample of the CC-CV charge.
	Returns:
		dict: A dictionary with keys ['cell_id', 'group_id', 'rpt', 'num_cycles', 'voltage', 'q_dchg', 'dcir_chg_20', 'dcir_chg_50', 'dcir_chg_90', 'dcir_dchg_20', 'dcir_dchg_50', 'dcir_dchg_90']
	"""

	cc_data = {		
		'cell_id': [],		# the corresponding cell id of each element
		'group_id': [],		# the corresponding group id of each element
		'rpt':[],			# the corresponding rpt number of each element 
		'num_cycles':[], 	# the number of aging cycles at the current rpt number
		'voltage':[],		# each element is an array of voltage corresponding to a single subsample of the CC-CV charge
		'q_dchg':[],		# the corresponding remaining cell discharge capacity of each element
		'dcir_chg_20':[],	# the cell's internal charge resistance at 20% SOC
		'dcir_chg_50':[],	# the cell's internal charge resistance at 50% SOC
		'dcir_chg_90':[],	# the cell's internal charge resistance at 90% SOC
		'dcir_dchg_20':[],	# the cell's internal discharge resistance at 20% SOC
		'dcir_dchg_50':[],	# the cell's internal discharge resistance at 50% SOC
		'dcir_dchg_90':[],	# the cell's internal discharge resistance at 90% SOC
	}
	for cell_id in df_test_tracker['Cell ID'].unique():
		group_id = get_group_id(cell_id)
		rpt_data = load_preprocessed_data(get_preprocessed_data_files(dir_preprocessed_data, data_type='rpt', cell_id=cell_id))
		if rpt_data is None:
			raise ValueError(f"There is no preprocessed cycling data for cell {cell_id}. Please check that the preprocessed data is downloaded properly and filenames have not been altered.")
		
		# filter to only first life data
		rpt_data = rpt_data.loc[rpt_data['Life'] == '1st'].copy()
		
		# get health metrics for this cell
		df_health_features = get_health_features_from_rpt_data(rpt_data)

		# get cc-cv charge data from rpt data
		df_cc = extract_cccv_charge(rpt_data=rpt_data, plot_interpolation=False)
		
		#region: get information from each RPT (just the CC-CV charge data)
		for rpt_num,df_split in df_cc.sort_values(by=['RPT Number', 'Time (s)'], ascending=[True, True]).groupby('RPT Number'):
			#region: get q_dchg and cycle numbers for this rpt
			q_dchg = df_health_features.loc[df_health_features['rpt'] == rpt_num, 'q_dchg'].values[0]
			num_cycles = rpt_data.loc[rpt_data['RPT Number'] == rpt_num, 'Num Cycles'].max()
			#endregion

			#region: drop data that falls outside specified SOC bounds
			for b in soc_bounds: assert b >= 0.0 and b <= 1.0
			# lower voltage limit per soc_bounds[1] (inclusive)
			vlim_lower = df_v_vs_soc_1c_chg.loc[df_v_vs_soc_1c_chg['soc'] <= soc_bounds[0]*100, 'v'].values[-1]
			# upper voltage limit per soc_bounds[1] (exclusive)
			vlim_upper = df_v_vs_soc_1c_chg.loc[df_v_vs_soc_1c_chg['soc'] < soc_bounds[1]*100, 'v'].values[-1]
			# filter df to only within this voltage range
			df_split = df_split.loc[(df_split['Voltage (V)'] >= vlim_lower) & (df_split['Voltage (V)'] < vlim_upper)]
			#endregion

			# get CC subsamples from this full CC-CV charge signal
			v_samples = get_cc_subsamples(df_split['Voltage (V)'].values, segment_length=segment_length, segment_overlap=segment_overlap)
			
			# add other identifying info corresponding to this CC subsample
			for v in v_samples:
				if not len(v) == segment_length: continue
				cc_data['cell_id'].append(cell_id)
				cc_data['group_id'].append(group_id)
				cc_data['rpt'].append(rpt_num)
				cc_data['num_cycles'].append(num_cycles)
				cc_data['voltage'].append(v)
				cc_data['q_dchg'].append(q_dchg)
				cc_data['dcir_chg_20'].append(
					df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_chg_20'].values[0])
				cc_data['dcir_chg_50'].append(
					df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_chg_50'].values[0])
				cc_data['dcir_chg_90'].append(
					df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_chg_90'].values[0])
				cc_data['dcir_dchg_20'].append(
					df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_dchg_20'].values[0])
				cc_data['dcir_dchg_50'].append(
					df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_dchg_50'].values[0])
				cc_data['dcir_dchg_90'].append(
					df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_dchg_90'].values[0])
		#endregion


	#region: convert all dic values to np arrays
	cc_data['cell_id'] = np.asarray(cc_data['cell_id'])
	cc_data['group_id'] = np.asarray(cc_data['group_id'])
	cc_data['rpt'] = np.asarray(cc_data['rpt'])
	cc_data['num_cycles'] = np.asarray(cc_data['num_cycles'])
	cc_data['voltage'] = np.asarray(cc_data['voltage'])
	cc_data['q_dchg'] = np.asarray(cc_data['q_dchg'])
	cc_data['dcir_chg_20'] = np.asarray(cc_data['dcir_chg_20'])
	cc_data['dcir_chg_50'] = np.asarray(cc_data['dcir_chg_50'])
	cc_data['dcir_chg_90'] = np.asarray(cc_data['dcir_chg_90'])
	cc_data['dcir_dchg_20'] = np.asarray(cc_data['dcir_dchg_20'])
	cc_data['dcir_dchg_50'] = np.asarray(cc_data['dcir_dchg_50'])
	cc_data['dcir_dchg_90'] = np.asarray(cc_data['dcir_dchg_90'])
	#endregion
	return cc_data

def create_slowpulse_modeling_data(dir_preprocessed_data:Path) -> dict:
	"""Obtains all 1st-life information used for SOH estimation modeling using slow pulses.

	Args:
		dir_preprocessed_data (Path): location of downloaded preprocessed data
	Returns:
		dict: A dictionary with keys ['cell_id', 'group_id', 'rpt', 'soc', 'pulse_type', 'voltage', 'q_dchg', 'dcir_chg_20', 'dcir_chg_50', 'dcir_chg_90', 'dcir_dchg_20', 'dcir_dchg_50', 'dcir_dchg_90']
	"""

	pulse_data = {		
		'cell_id': [],		# the corresponding cell id of each element
		'group_id': [],		# the corresponding group id of each element
		'rpt':[],			# the corresponding rpt number of each element 
		'num_cycles':[],	# the number of aging cycles at the current rpt number
		'soc':[], 			# the corresponding SOC of each pulse {20,50,90}
		'soc - coulomb':[],
		'pulse_type':[],	# the corresponding pulse type {'chg', 'dchg'}
		'voltage':[],		# each element is an array of voltage corresponding to a single slow pulse
		'q_dchg':[],		# the corresponding remaining cell discharge capacity of each element
		'dcir_chg_20':[],	# the cell's internal charge resistance at 20% SOC
		'dcir_chg_50':[],	# the cell's internal charge resistance at 50% SOC
		'dcir_chg_90':[],	# the cell's internal charge resistance at 90% SOC
		'dcir_dchg_20':[],	# the cell's internal discharge resistance at 20% SOC
		'dcir_dchg_50':[],	# the cell's internal discharge resistance at 50% SOC
		'dcir_dchg_90':[],	# the cell's internal discharge resistance at 90% SOC
	}
	
	for cell_id in df_test_tracker['Cell ID'].unique():
		group_id = get_group_id(cell_id)
		rpt_data = load_preprocessed_data(get_preprocessed_data_files(dir_preprocessed_data, data_type='rpt', cell_id=cell_id))
		if rpt_data is None:
			raise ValueError(f"There is no preprocessed cycling data for cell {cell_id}. Please check that the preprocessed data is downloaded properly and filenames have not been altered.")
	
		# filter to only first life data
		rpt_data = rpt_data.loc[rpt_data['Life'] == '1st'].copy()

		# get health metrics for this cell
		df_health_features = get_health_features_from_rpt_data(rpt_data)
		
		#region: add cumulative capacity column (for coulomb counting)
		q_cums = np.zeros(len(rpt_data))
		for rpt_num in rpt_data['RPT Number'].unique():
			df_rpt = rpt_data.loc[rpt_data['RPT Number'] == rpt_num]
			dqs = np.zeros(len(df_rpt))
			for step_num in df_rpt['Step Number'].unique():
				df_step = df_rpt.loc[(df_rpt['Step Number'] == step_num)]
				idxs = df_step.index.values - df_rpt.index.values[0]
				
				assert len(df_step['State'].unique()) == 1
				sign = -1 if 'DChg' in df_step['State'].unique()[0] else 1

				dqs[idxs[1:]] = df_step['Capacity (Ah)'].diff().values[1:] * sign
				dqs[idxs[0]] = 0

			# # need to offset q_cum to start at first charge step after discharge 
			# offset_idx = df_rpt.loc[df_rpt['Step Number'] == 2].index.values[0] - df_rpt.index.values[0]
			q_cum = np.cumsum(dqs)
			q_cums[df_rpt.index.values] = q_cum - q_cum[-1] # q_cum[offset_idx]

		rpt_data['Cumulative Capacity (Ahr)'] = q_cums
		#endregion

		# get information from each RPT 
		for rpt_num,df_split in rpt_data.sort_values(by=['RPT Number', 'Time (s)'], ascending=[True, True]).groupby('RPT Number'):
			#region: get q_dchg and cycle numbers for this rpt
			q_dchg = df_health_features.loc[df_health_features['rpt'] == rpt_num, 'q_dchg'].values[0]
			num_cycles = df_split['Num Cycles'].max()
			#endregion

			# get pulse data for each pulse type and soc combination
			for soc in [20, 50, 90]:
				for p_type in ['chg', 'dchg']:
					# filter dataframe to current slow pulse (using RPT number, SOC, pulse type)
					df_pulse = df_split.loc[(df_split['Segment Key'] == 'slowpulse') & \
											(df_split['Pulse SOC'] == soc) & \
											(df_split['Pulse Type'] == p_type)]

					#region: interpolate pulse voltage to exactly 100 seconds 
					# each segment of pulse should be the following lengths (in seconds)
					seg_lengths = [30,10,60]
					seg_ts = []
					seg_vs = []
					for i, step in enumerate(df_pulse['Step Number'].unique()):
						df_seg = df_pulse.loc[df_pulse['Step Number'] == step]
						t = df_seg['Time (s)'].values - df_seg['Time (s)'].values[0]
						t_interp = np.arange(0, seg_lengths[i], 1)
						f_v = interpolate.PchipInterpolator(t, df_seg['Voltage (V)'].values)
						v_interp = f_v(t_interp)
						seg_ts.append(t_interp)
						seg_vs.append(v_interp)
		
					pulse_v = np.hstack(seg_vs)
					assert len(pulse_v) == 100
					#endregion

					# add values to dictionary
					pulse_data['cell_id'].append(cell_id)
					pulse_data['group_id'].append(group_id)
					pulse_data['rpt'].append(rpt_num)
					pulse_data['num_cycles'].append(num_cycles)
					pulse_data['soc'].append(soc)
					pulse_data['soc - coulomb'].append( df_pulse['Cumulative Capacity (Ahr)'].values[0] / q_dchg * 100 )
					pulse_data['pulse_type'].append(p_type)
					pulse_data['voltage'].append(pulse_v)
					pulse_data['q_dchg'].append(q_dchg)

					pulse_data['dcir_chg_20'].append(
						df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_chg_20'].values[0])
					pulse_data['dcir_chg_50'].append(
						df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_chg_50'].values[0])
					pulse_data['dcir_chg_90'].append(
						df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_chg_90'].values[0])
					pulse_data['dcir_dchg_20'].append(
						df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_dchg_20'].values[0])
					pulse_data['dcir_dchg_50'].append(
						df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_dchg_50'].values[0])
					pulse_data['dcir_dchg_90'].append(
						df_health_features.loc[df_health_features['rpt'] == rpt_num, 'dcir_dchg_90'].values[0])

	#region: convert all dic values to np arrays
	pulse_data['cell_id'] = np.asarray(pulse_data['cell_id'])
	pulse_data['group_id'] = np.asarray(pulse_data['group_id'])
	pulse_data['rpt'] = np.asarray(pulse_data['rpt'])
	pulse_data['num_cycles'] = np.asarray(pulse_data['num_cycles'])
	pulse_data['soc'] = np.asarray(pulse_data['soc'])
	pulse_data['soc - coulomb'] = np.asarray(pulse_data['soc - coulomb'])
	pulse_data['pulse_type'] = np.asarray(pulse_data['pulse_type'])
	pulse_data['voltage'] = np.asarray(pulse_data['voltage'])
	pulse_data['q_dchg'] = np.asarray(pulse_data['q_dchg'])
	pulse_data['dcir_chg_20'] = np.asarray(pulse_data['dcir_chg_20'])
	pulse_data['dcir_chg_50'] = np.asarray(pulse_data['dcir_chg_50'])
	pulse_data['dcir_chg_90'] = np.asarray(pulse_data['dcir_chg_90'])
	pulse_data['dcir_dchg_20'] = np.asarray(pulse_data['dcir_dchg_20'])
	pulse_data['dcir_dchg_50'] = np.asarray(pulse_data['dcir_dchg_50'])
	pulse_data['dcir_dchg_90'] = np.asarray(pulse_data['dcir_dchg_90'])
	#endregion

	return pulse_data

def save_modeling_data(data_type:str, data:dict, filename:str=None) -> Path:
	"""Saves generated modeling data using a pre-set organizational method

	Args:
		data_type (str): {'cc', 'slowpulse', 'fastpulse', 'ultrafastpulse'}. The data_type the data corresponds to.
		data (dict): The data to be saved. Should be in the standard dictionary format
		filename (str, optional): Can optionally specify the filename. If not provided, an auto-indexing naming convention is used. Defaults to None.
	
	Returns:
		Path: the Path object to where the data was saved
	"""

	assert data_type in ['cc', 'slowpulse', 'fastpulse', 'ultrafastpulse']

	# may need to make the directory if first time being run
	if not dir_processed_data.joinpath(data_type).exists():
		dir_processed_data.joinpath(data_type).mkdir(exist_ok=True, parents=True)

	#region: use auto-naming convention if filename wasn't specified
	file_idx = None
	if filename is None:
		prev_files = sorted(dir_processed_data.joinpath(data_type).glob(f"processed_data_{data_type}*"))
		if len(prev_files) == 0: 
			file_idx = 0
		else:
			split_start_idx = str(prev_files[-1]).rindex(f'{data_type}_') + len(f'{data_type}_')
			split_stop_idx = str(prev_files[-1]).rindex('.pkl')
			file_idx = int(str(prev_files[-1])[split_start_idx:split_stop_idx]) + 1
		filename = f"processed_data_{data_type}_{file_idx}.pkl"
	#endregion

	# save data and return path to where data was saved
	f = dir_processed_data.joinpath(data_type, filename)
	pickle.dump(data, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	return f


def perform_postprocessing(dir_preprocessed_data:Path):
	print("Creating dataset for CC-segment model...")
	cc_data = create_cc_modeling_data(dir_preprocessed_data=dir_preprocessed_data, 
                                      segment_length=600, segment_overlap=0.5, soc_bounds=(0.3,0.9))
	x = save_modeling_data(data_type='cc', data=cc_data)
	print(f"CC-segment dataset saved to: {x}")
	
	print("Creating dataset for slowpulse model...")
	pulse_data = create_slowpulse_modeling_data(dir_preprocessed_data)
	x = save_modeling_data(data_type='slowpulse', data=pulse_data)
	print(f"Slowpulse dataset saved to: {x}")
	

if __name__ == '__main__':
    # if using external SSD
	temp = Path("/Volumes/T7/Datasets to Publish/ILCC-LFP-aging-dataset")
	perform_postprocessing(dir_preprocessed_data=temp)
	
	print('postprocessing.py complete.\n')
