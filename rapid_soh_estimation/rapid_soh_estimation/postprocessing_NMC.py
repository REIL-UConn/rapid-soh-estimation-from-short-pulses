
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from common_methods import *

import zipfile


def get_empty_cell_dic():
    dic = {'cell_id':None, 'rpt_date':[], 'rpt_tracker':[]}
    
    # set pulse keys
    for pulse_type in ['chg', 'chgRest', 'dchg', 'dchgRest']:
        for soc in [30, 50, 70]:
            for crate in ['0C25', '0C50', '1C00', '2C00', '3C00']:
                for suffix in ['v', 'i', 't', 'T', 'dcir']:
                    dic[f"pulse_{pulse_type}_{soc}soc_{crate}_{suffix}"] = []
    
    # set full charge/discharge keys
    for pulse_type in ['chg', 'dchg']:
        for suffix in ['v', 'i', 't', 'T', 'q', 'qmax']:
            dic[f"cc_{pulse_type}_1C00_{suffix}"] = []
                
    return dic

def get_cell_id_from_filename(filename:str):
    x = filename.index('ep sanyo') + 9
    return int(filename[x:x+3])

def get_date_from_filename(filename:str):
    x = filename.index('Massenalterung=') + len('Massenalterung=')
    y = filename[x:].index('=')
    date_str = filename[x:x+y]
    date = pd.to_datetime(date_str, format="%Y-%m-%d %H%M%S")
    return date

def file_contains_valid_protocol(filename:str):
    try:
        x = filename.index('TBA')
        return True
    except ValueError:
        return False
    
def get_protocol_from_filename(filename:str):
    x = filename.index('TBA')
    y = filename[x:].index('=')
    return filename[x:x+y]

def get_filename_keys(filename:str):
    '''
    returns cell_id, date, protocol_type from provide filename
    '''
    return get_cell_id_from_filename(filename), get_date_from_filename(filename), get_protocol_from_filename(filename)

def get_cumulative_capacity(raw_qs):
    delta_q = np.diff(raw_qs)
    delta_q[delta_q < 0] = 0.0
    cum_q = np.zeros_like(raw_qs)
    for i, dq in enumerate(delta_q):
        cum_q[i+1] = cum_q[i] + dq
    return cum_q
        
def process_raw_data(overwrite_existing=False):
    reprocess_all_data = overwrite_existing
    f_processed_data_all = dir_processed_data.joinpath("NMC", "All")
    f_processed_data_all.mkdir(parents=True, exist_ok=True)

    #region: get list of cells that have already been processed
    f_already_processed = f_processed_data_all.glob('*.pkl')
    cell_ids_processed = []
    for f in f_already_processed:
        x = f.name.index('Sanyo ') + len('Sanyo ')
        cell_ids_processed.append( int(f.name[x:x+3]) )
    #endregion

    f_rawdata = dir_NMC_data.joinpath("Rohdaten")
    f_all_zip = f_rawdata.glob("*.zip")
    file_dic = {}       # keys are the types of protocols (each entry contains the cell id, date, and filepath)
    for f in f_all_zip:
        if file_contains_valid_protocol(f.name):
            cell_id, date, protocol_type = get_filename_keys(f.name)
            if protocol_type not in list(file_dic.keys()):
                file_dic[protocol_type] = {'cell_id':np.array([]), 'date':np.array([]), 'filepath':np.array([])}
                
            if (cell_id not in cell_ids_processed) or (reprocess_all_data):
                file_dic[protocol_type]['cell_id'] = np.append(file_dic[protocol_type]['cell_id'], cell_id)
                file_dic[protocol_type]['date'] = np.append(file_dic[protocol_type]['date'], date)
                file_dic[protocol_type]['filepath'] = np.append(file_dic[protocol_type]['filepath'], f)

    # need to group RPT files by cell id and then sort by date
    rpt_key = 'TBA_CUv2'
    f_unzip_folder = dir_NMC_data.joinpath("Unzipped Data")
    f_unzip_folder.mkdir(parents=True, exist_ok=True)

    for cell_id in np.unique(file_dic[rpt_key]['cell_id']):
        print(f"Processing Cell {cell_id}...")
        idxs  = np.where(file_dic[rpt_key]['cell_id'] == cell_id)
        sort_idxs = file_dic[rpt_key]['date'][idxs].argsort()
        
        cell_dic = get_empty_cell_dic()
        cell_dic['cell_id'] = cell_id
        
        for i in range(len(sort_idxs)):
            cell_dic['rpt_date'].append( file_dic[rpt_key]['date'][idxs][sort_idxs][i] )
            cell_dic['rpt_tracker'].append(i)
            
            # unzip file & load csv
            f_zip = file_dic[rpt_key]['filepath'][idxs][sort_idxs][i]
            with zipfile.ZipFile(f_zip, 'r') as zObject: 
                # Extracting all zip into new filepath
                zObject.extractall(path=f_unzip_folder)
                
            f_unzipped = f_unzip_folder.joinpath(f'{f_zip.stem}.csv')
            data = pd.read_csv(f_unzipped)
        
            #region: info on data columns
            # All unique keys in "data['Prozedur']"  --> 'Prozedur' = step type
            #   TBA_CUv2        initial rest
            #   TBA_SD          CC discharge at 1C
            #   TBA_SC          CC-CV charge at 1C
            #   TBA_WS          rest period
            #   TBA_SCAPv2      CC-CV discharge at 1C and 0.5C
            #   TBA_SQOCVv2     0.25C CC-CV dchg & chg
            #   TBA_SOCV        CC-CV discharge at 1C to each SOC level
            #   TBA_PULS        pulse data
            #   TBA_SOC         rest periods in pulse data
            # All unique keys in "data['Zustand']"  --> 'Zustand' = tester mode (CHA=chg, DCH=dchg, PAU=rest, STO=stop)
            #endregion

            temperature_key = None
            for col in data.columns.values:
                if 'Temp' in col:
                    temperature_key = col
                
            #region: Extract 1C CC-CV charge
            ti = data.loc[(data['Prozedur'] == 'TBA_SC') & (data['Schrittdauer'] == 0), 'Programmdauer'].iloc[4]
            tf = data.loc[(data['Prozedur'] == 'TBA_SC') & (data['Schrittdauer'] == 0), 'Programmdauer'].iloc[6]
            df_temp = data.loc[(data['Prozedur'] == 'TBA_SC')]
            df_filt = df_temp.loc[(df_temp['Programmdauer'] >= ti) & (df_temp['Programmdauer'] < tf)]

            cell_dic['cc_chg_1C00_t'].append(df_filt['Programmdauer'].values.astype(float) / 1000)
            cell_dic['cc_chg_1C00_t'][-1] -= cell_dic['cc_chg_1C00_t'][-1][0]
            cell_dic['cc_chg_1C00_v'].append(df_filt['Spannung'].values.astype(float))
            cell_dic['cc_chg_1C00_i'].append(df_filt['Strom'].values.astype(float))
            cell_dic['cc_chg_1C00_T'].append(df_filt[temperature_key].values.astype(float))
            cell_dic['cc_chg_1C00_q'].append(get_cumulative_capacity(df_filt['AhStep'].values.astype(float)))
            cell_dic['cc_chg_1C00_qmax'].append(np.max(cell_dic['cc_chg_1C00_q'][-1]).astype(float))
            #endregion
            
            #region: Extract 1C CC-CV discharge
            ti = data.loc[(data['Prozedur'] == 'TBA_SCAPv2') & (data['Schrittdauer'] == 0), 'Programmdauer'].iloc[0]
            tf = data.loc[(data['Prozedur'] == 'TBA_SCAPv2') & (data['Schrittdauer'] == 0), 'Programmdauer'].iloc[2]
            df_temp = data.loc[(data['Prozedur'] == 'TBA_SCAPv2')]
            df_filt = df_temp.loc[(df_temp['Programmdauer'] >= ti) & (df_temp['Programmdauer'] < tf)]

            cell_dic['cc_dchg_1C00_t'].append(df_filt['Programmdauer'].values.astype(float) / 1000)
            cell_dic['cc_dchg_1C00_t'][-1] -= cell_dic['cc_dchg_1C00_t'][-1][0]
            
            cell_dic['cc_dchg_1C00_v'].append(df_filt['Spannung'].values.astype(float))
            cell_dic['cc_dchg_1C00_i'].append(df_filt['Strom'].values.astype(float))
            cell_dic['cc_dchg_1C00_T'].append(df_filt[temperature_key].values.astype(float))
            cell_dic['cc_dchg_1C00_q'].append(get_cumulative_capacity(df_filt['AhStep'].values.astype(float)))
            cell_dic['cc_dchg_1C00_qmax'].append(np.max(cell_dic['cc_dchg_1C00_q'][-1]).astype(float))
            #endregion

            #region: Extract pulse data
            crate_step_dic = {      # corrsponding step index ('Schritt') for each portion of each DC pulse
                '0C25': {'chg':4,  'chgRest':5,  'dchg':8,  'dchgRest':9},
                '0C50': {'chg':14, 'chgRest':15, 'dchg':18, 'dchgRest':19},
                '1C00': {'chg':24, 'chgRest':25, 'dchg':28, 'dchgRest':29},
                '2C00': {'chg':34, 'chgRest':35, 'dchg':38, 'dchgRest':39},
                '3C00': {'chg':44, 'chgRest':45, 'dchg':48, 'dchgRest':49},
            }
            for i, soc in enumerate([70,50,30]):
                for crate in crate_step_dic.keys():
                    for pulse_type in crate_step_dic[crate].keys():
                        step_idx = crate_step_dic[crate][pulse_type]
                        df_temp = data.loc[(data['Prozedur'] == 'TBA_PULS') & \
                                        (data['Schrittdauer'] == 0) & \
                                        (data['Schritt'] == step_idx)]
                        # some files have steps that don't begin at zero seconds, need to adjust filtering process
                        if len(df_temp) < 3:
                            df_temp = data.loc[(data['Prozedur'] == 'TBA_PULS') & \
                                            (data['Schritt'].values.astype(float) == step_idx)]
                            
                            # add difference in step time as new column
                            diff_Sc = np.zeros(len(df_temp))
                            diff_Sc[1:] = np.diff(df_temp['Schrittdauer'].values.astype(float))
                            df_temp.insert(0, 'Diff_Sc', diff_Sc)
                            
                            # add difference in program time as new column
                            diff_Pr = np.zeros(len(df_temp))
                            diff_Pr[1:] = np.diff(df_temp['Programmdauer'].values.astype(float))
                            df_temp.insert(0, 'Diff_Pr', diff_Pr)
                        
                            # new filtering -> find where two times diffs don't match or step time = 0
                            init_row = df_temp.iloc[0:1]
                            df_temp = df_temp.loc[((df_temp['Diff_Pr'] - df_temp['Diff_Sc']) != 0) | \
                                                (df_temp['Schrittdauer'] == 0)]
                            if len(df_temp) < 3:
                                df_temp = pd.concat([init_row, df_temp])
                            
                            assert len(df_temp) == 3, f"Failed to find start steps\ni:{i} step:{step_idx} file:{f_unzipped}"
        
                        df_filt = None
                        ti = df_temp['Programmdauer'].iloc[i]
                        try:
                            tf = df_temp['Programmdauer'].iloc[i+1]
                            df_filt = data.loc[(data['Prozedur'] == 'TBA_PULS') & \
                                            (data['Schritt'] == step_idx) & \
                                            (data['Programmdauer'] >= ti) & \
                                            (data['Programmdauer'] < tf)]
                        except IndexError:
                            df_filt = data.loc[(data['Prozedur'] == 'TBA_PULS') & \
                                            (data['Schritt'] == step_idx) & \
                                            (data['Programmdauer'] >= ti)]
                                    
                        key = f"pulse_{pulse_type}_{soc}soc_{crate}"
                        cell_dic[f'{key}_t'].append(df_filt['Programmdauer'].values.astype(float) / 1000)
                        cell_dic[f'{key}_t'][-1] -= cell_dic[f'{key}_t'][-1][0]
                        cell_dic[f'{key}_v'].append(df_filt['Spannung'].values.astype(float))
                        cell_dic[f'{key}_i'].append(df_filt['Strom'].values.astype(float))
                        cell_dic[f'{key}_T'].append(df_filt[temperature_key].values.astype(float))
                        dcir = None
                        if pulse_type == 'chg' or pulse_type == 'dchg':
                            v_i = data.loc[data['Programmdauer'] < ti, 'Spannung'].values.astype(float)[-1]   # initial voltage
                            i_i = data.loc[data['Programmdauer'] < ti, 'Strom'].values.astype(float)[-1]      # initial current
                            v_f = np.average(cell_dic[f'{key}_v'][-1][-5:]) # averaging last 5 points due to noise
                            i_f = np.average(cell_dic[f'{key}_i'][-1][-5:]) # averaging last 5 points due to noise
                            dcir = (v_f - v_i) / (i_f - i_i)
                        cell_dic[f'{key}_dcir'].append(dcir)
            #endregion
        
        # save cell_dic to pickle file
        f_save = f_processed_data_all.joinpath("Sanyo {:03d}.pkl".format(cell_dic['cell_id'].astype(int)))
        with open(f_save, 'wb') as handle:
            pickle.dump(cell_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print()
        
    print("Processing complete.")



def load_NMC_data(filepath):
    f_all_cells = filepath.glob("*.pkl")
    dataset = {}
    for f in f_all_cells:
        cell_data = pickle.load(open(f, 'rb'))
        dataset[int(cell_data['cell_id'])] = cell_data
        
    return dataset

def interp_time_series(t_arr, y_arrs, n_points):
    'returns (t_arr, y_arrs) interpolated to n_points'
    
    from scipy import interpolate
    t_clean, clean_idxs = np.unique(t_arr, return_index=True)

    t_interp = []
    y_arrs_interp = []
    for y_arr in y_arrs:
        assert len(t_arr) == len(y_arr)
        y_clean = y_arr[clean_idxs]
        assert len(t_clean) == len(y_clean)
        f = interpolate.PchipInterpolator(t_clean, y_clean)
        t_interp = np.linspace(t_arr[0], t_arr[-1], n_points)
        y_arrs_interp.append( f(t_interp) )
    return t_interp, y_arrs_interp

def find_CV_pulses(all_pulses_v, pulse_type='chg'):
    '''
    Returns indices of pulses in array that hit voltage limit
    all_pulses_v: array of time-series pulse voltages
    pulse_type: 'chg' or 'dchg'
    '''
    assert pulse_type in ['chg', 'dchg']
    tol = 0.005
    idxs = []
    for i, vs in enumerate(all_pulses_v):
        if pulse_type == 'chg':
            if np.max(vs) >= 4.1 - tol:
                idxs.append(i)
        elif pulse_type == 'dchg':
            if np.min(vs) <= 3.0 + tol:
                idxs.append(i)
    return idxs

def is_clean_pulse(voltage, pulse_type='chg', tol=0.005):
    '''
    Returns true is pulse does not hit voltage limit (ie no CV step present)
    voltage: pulse voltage
    pulse_type: 'chg' or 'dchg'
    tol: voltage tolerance on limits (3.0, 4.1) V
    '''
    assert pulse_type in ['chg', 'dchg']
    tol = 0.005
    
    if pulse_type == 'chg':
        return np.max(voltage) < 4.1 - tol
    elif pulse_type == 'dchg':
        return np.min(voltage) > 3.0 + tol
    
def find_CC_segment_from_full(full_v, cc_type='chg', tol=0.010):
    '''
    Returns indices of passed full charge or discharge voltage corresponding to only the CC segment
    full_v: voltage of a single full CC-CV charge or discharge sequence
    cc_type: 'chg' or 'dchg'
    tol: tolerance on voltage limit (eg. for chg, return idxs when voltage < (4.1-tol))
    '''
    assert cc_type in ['chg', 'dchg']
    idxs = []
    for i in range(len(full_v)):
        if cc_type == 'chg':
            if full_v[i] < 4.1 - tol:
                idxs.append(i)
            else:
                return idxs
        elif cc_type == 'dchg':
            if full_v[i] > 3.0 + tol:
                idxs.append(i)
            else:
                return idxs
    return idxs     

def get_cc_subsamples(voltage, time, seg_length, sampling_rate=1, seg_overlap=0.5):
    '''
    voltage: voltage array
    time: time array corresponding to voltage
    seg_length: time length of each subsample (in seconds)
    sampling_rate: sample per second to interpolate segment to 
    seg_overlap: percent each new segment can overlap with previous segment
    Returns: (v_segments, t_segments)
    '''
    
    v_segments = []
    t_segments = []
    
    segment_start_idx = 0
    segment_end_idx = np.where(time >= time[segment_start_idx] + seg_length)[0][0]    
    while segment_end_idx < len(time):
        t_segment = time[segment_start_idx:segment_end_idx]
        v_segment = voltage[segment_start_idx:segment_end_idx]
        
        # interp segment to specified sample size
        t_interp, y_arrs_interp = interp_time_series(t_segment, [v_segment], (seg_length * sampling_rate))
        v_interp = y_arrs_interp[0]
        
        v_segments.append( v_interp )
        t_segments.append( t_interp )
        segment_start_idx = int(segment_end_idx - ((segment_end_idx - segment_start_idx) * seg_overlap)) + 1
        try:
            segment_end_idx = np.where(time >= (time[segment_start_idx] + seg_length))[0][0]
        except IndexError:
            break
        
    return np.array(v_segments), np.array(t_segments)



def get_dataset_for_cc_modeling(input_length=10):
    '''
    input_length: length (in minutes) of cc segment to use
    '''
    
    f_processed_data_all = dir_processed_data.joinpath("NMC", "All")
    dataset = load_NMC_data(f_processed_data_all)
    assert len(dataset.keys()) > 0
    
    modeling_dic = {'model_input':[], 'model_input_times':[], 'model_output':[], 'model_cell_id':[], 'model_soc':[]}
    modeling_dic['dataset_dic'] = dataset

    model_input_CC_length = input_length*60     # length of CC segment to use (in seconds)
    model_input_CC_overlap = 0              # % overlap of CC segments when creating subsamples (0 = no overlap) --> must be [0, 1.0)
    model_input_samples_per_second = 1      # segment will be interpolated using this value
    model_input_soc_bounds = (30,90)        # will pull random CC segment between these SOC bounds
    nominal_q0 = 1.80                       # all cells start at 1.80 Ah +/- 0.05
    
    for cell_id_key in dataset.keys():
        # filter dataset to only 1st life
        idxs_1stLife = np.where(np.asarray(dataset[cell_id_key]['cc_dchg_1C00_qmax']) >= nominal_q0 * 0.80)[0]
        for i in idxs_1stLife:
            #region: get outputs corresponding to this RPT
            output = [ 
                dataset[cell_id_key]['cc_dchg_1C00_qmax'][i],
                
                dataset[cell_id_key]['pulse_chg_30soc_1C00_dcir'][i],
                dataset[cell_id_key]['pulse_chg_50soc_1C00_dcir'][i],
                dataset[cell_id_key]['pulse_chg_70soc_1C00_dcir'][i],
                
                dataset[cell_id_key]['pulse_dchg_30soc_1C00_dcir'][i],
                dataset[cell_id_key]['pulse_dchg_50soc_1C00_dcir'][i],
                dataset[cell_id_key]['pulse_dchg_70soc_1C00_dcir'][i],
            ]
            #endregion
            
            #region: clean arrays to only CC portion
            cc_idxs = find_CC_segment_from_full(dataset[cell_id_key]['cc_chg_1C00_v'][i], cc_type='chg', tol=0.01)
            if len(cc_idxs) == 0:
                # if sequence at current RPT doesn't contain anny CC section, skip (ssome data has errors, need to skip)
                continue
            #endregion
            cc_time = np.asarray(dataset[cell_id_key]['cc_chg_1C00_t'][i])[cc_idxs]
            cc_voltage = np.asarray(dataset[cell_id_key]['cc_chg_1C00_v'][i])[cc_idxs]
            cc_capacity = np.asarray(dataset[cell_id_key]['cc_chg_1C00_q'][i])[cc_idxs]
            qmax = dataset[cell_id_key]['cc_chg_1C00_qmax'][i]
            
            #region: filter to limited SOC range
            lb = model_input_soc_bounds[0] / 100
            ub = model_input_soc_bounds[1] / 100
            idx_under_lb = np.where(cc_capacity <= lb * qmax)[0][-1]
            idx_under_ub = np.where(cc_capacity <= ub * qmax)[0][-1]
            #endregion
            idxs_inSOCBounds = np.r_[idx_under_lb:idx_under_ub]
            voltage_all = cc_voltage[idxs_inSOCBounds]
            time_all = cc_time[idxs_inSOCBounds]
            
            # create sub-samples from voltage & time
            v_segments, t_segments = get_cc_subsamples(voltage_all, time_all, model_input_CC_length, 
                                                    sampling_rate=model_input_samples_per_second,
                                                    seg_overlap=model_input_CC_overlap)
            for i in range(len(v_segments)):
                start_idx = np.argwhere(cc_voltage >= v_segments[i][0])[0][0]      # idx of segment in full voltage sequence
                soc = cc_capacity[start_idx] / qmax * 100
                modeling_dic['model_soc'].append( soc )
                modeling_dic['model_input'].append( v_segments[i] )
                modeling_dic['model_input_times'].append( t_segments[i] )
                modeling_dic['model_output'].append(output)
                modeling_dic['model_cell_id'].append(int(cell_id_key))
    
                
    # create scalers and scaled data
    from sklearn.preprocessing import StandardScaler
    modeling_dic['model_input_scaler'] = StandardScaler()
    modeling_dic['model_input_scaled'] = modeling_dic['model_input_scaler'].fit_transform(modeling_dic['model_input'])
    modeling_dic['model_output_scaler'] = StandardScaler()
    modeling_dic['model_output_scaled'] = modeling_dic['model_output_scaler'].fit_transform(modeling_dic['model_output'])
    
    # format arrays
    modeling_dic['model_output'] = np.array(modeling_dic['model_output'])
    modeling_dic['model_input'] = np.array(modeling_dic['model_input'])
    modeling_dic['model_input_times'] = np.array(modeling_dic['model_input_times'])
    modeling_dic['model_cell_id'] = np.array(modeling_dic['model_cell_id'])
    modeling_dic['model_soc'] = np.array(modeling_dic['model_soc'])
    
    return modeling_dic

def get_dataset_for_pulse_modeling(pulse_type='chg', pulse_soc=30, c_rate=1, use_relative_voltage=False, remove_CV=True):
    assert pulse_soc in [30,50,70,None]
    assert pulse_type in ['chg', 'dchg']
    assert c_rate in [0.25, 0.5, 1, 2, 3]
    
    f_processed_data_all = dir_processed_data.joinpath("NMC", "All")
    dataset = load_NMC_data(f_processed_data_all)
    assert len(dataset.keys()) > 0
    modeling_dic = {'model_input':[], 'model_input_times':[], 'model_output':[], 'model_cell_id':[], 'model_soc':[]}
    modeling_dic['dataset_dic'] = dataset

    for soc in ([pulse_soc] if pulse_soc is not None else [30,50,70]):
        c_rate_key = "{:01d}C{:02d}".format(np.floor(c_rate).astype(int), np.floor((c_rate-np.floor(c_rate).astype(int))*100).astype(int))
        pulse_key = f"pulse_{pulse_type}_{soc}soc_{c_rate_key}"
        rest_key = f"pulse_{pulse_type}Rest_{soc}soc_{c_rate_key}"
        nominal_q0 = 1.80                       # all cells start at 1.80 Ah +/- 0.05    
        for cell_id_key in dataset.keys():

            # filter dataset to only 1st life
            idxs_1stLife = np.where(np.asarray(dataset[cell_id_key]['cc_dchg_1C00_qmax']) >= nominal_q0 * 0.80)[0]
            for i in idxs_1stLife:
                #region: get outputs corresponding to this RPT
                output = [ 
                    dataset[cell_id_key]['cc_dchg_1C00_qmax'][i],
                    
                    dataset[cell_id_key]['pulse_chg_30soc_1C00_dcir'][i],
                    dataset[cell_id_key]['pulse_chg_50soc_1C00_dcir'][i],
                    dataset[cell_id_key]['pulse_chg_70soc_1C00_dcir'][i],
                    
                    dataset[cell_id_key]['pulse_dchg_30soc_1C00_dcir'][i],
                    dataset[cell_id_key]['pulse_dchg_50soc_1C00_dcir'][i],
                    dataset[cell_id_key]['pulse_dchg_70soc_1C00_dcir'][i],
                ]
                #endregion
                
                # interpolate pulse chg/dchg step (20 seconds)
                pulse_vs = dataset[cell_id_key][f"{pulse_key}_v"][i]
                pulse_ts = dataset[cell_id_key][f"{pulse_key}_t"][i]
                t_interp, y_arrs_interp = interp_time_series(pulse_ts, [pulse_vs], 20)
                pulse_ts = t_interp
                pulse_vs = y_arrs_interp[0]
                
                # interpolate rest step after pulse (30 seconds)
                rest_vs = dataset[cell_id_key][f"{rest_key}_v"][i]
                rest_ts = dataset[cell_id_key][f"{rest_key}_t"][i]
                t_interp, y_arrs_interp = interp_time_series(rest_ts, [rest_vs], 30)
                rest_ts = t_interp
                rest_vs = y_arrs_interp[0]
                
                all_v = np.hstack([pulse_vs, rest_vs])
                
                if not (remove_CV and not is_clean_pulse(all_v)):
                    if use_relative_voltage:
                        rel_v = all_v - all_v[0]
                        modeling_dic['model_input'].append( rel_v )
                    else: 
                        modeling_dic['model_input'].append( all_v )
                    modeling_dic['model_input_times'].append( np.hstack([pulse_ts, (rest_ts)+20]) )
                    
                    modeling_dic['model_output'].append( output )
                    modeling_dic['model_cell_id'].append( int(cell_id_key) )
                    modeling_dic['model_soc'].append( soc )
            
    if remove_CV and len(modeling_dic['model_input']) == 0:
        raise ValueError("The current pulse type does not have any pulses that don't hit the CV limit. Use a different pulse configuration or set \'remove_CV\' to False")
    
    # format arrays
    modeling_dic['model_output'] = np.array(modeling_dic['model_output'])
    modeling_dic['model_input'] = np.array(modeling_dic['model_input'])
    modeling_dic['model_input_times'] = np.array(modeling_dic['model_input_times'])
    modeling_dic['model_cell_id'] = np.array(modeling_dic['model_cell_id'])
    modeling_dic['model_soc'] = np.array(modeling_dic['model_soc'])

    # create scalers and scaled data
    from sklearn.preprocessing import StandardScaler
    modeling_dic['model_input_scaler'] = StandardScaler()
    modeling_dic['model_input_scaled'] = modeling_dic['model_input_scaler'].fit_transform(modeling_dic['model_input'])
    modeling_dic['model_output_scaler'] = StandardScaler()
    modeling_dic['model_output_scaled'] = modeling_dic['model_output_scaler'].fit_transform(modeling_dic['model_output'])
    
    return modeling_dic

def save_modeling_data(data_type:str, data:dict, filename:str=None) -> Path:
	"""Saves generated modeling data using a pre-set organizational method

	Args:
		data_type (str): {'cc', 'slowpulse'}. The data_type the data corresponds to.
		data (dict): The data to be saved. Should be in the standard dictionary format
		filename (str, optional): Can optionally specify the filename. If not provided, an auto-indexing naming convention is used. Defaults to None.
	
	Returns:
		Path: the Path object to where the data was saved
	"""

	assert data_type in ['cc', 'slowpulse']

	# may need to make the directory if first time being run
	if not dir_processed_data.joinpath("NMC", data_type).exists():
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
	else:
		if not filename[-4:] == '.pkl': filename += '.pkl'
	#endregion

	# save data and return path to where data was saved
	f = dir_processed_data.joinpath("NMC", data_type, filename)
	pickle.dump(data, open(f, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	return f

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
		f = dir_processed_data.joinpath("NMC", data_type, filename)
		if not f.exists(): 
			raise ValueError(f"Could not find specified file: {f}")
		return pickle.load(open(f, 'rb'))
	else:
		prev_files = sorted(dir_processed_data.joinpath("NMC", data_type).glob(f"processed_data_{data_type}_*"))
		if len(prev_files) == 0: 
			raise ValueError("Could not find any previously saved files. Try providing a filename")
		else:
			return pickle.load(open(prev_files[-1], 'rb'))



if __name__ == '__main__':
    process_raw_data(overwrite_existing=False)
    print("postprocessing_NMC.py complete")