
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from common_methods import *



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
                                (rpt_data['Step Number'] == 54)]
        
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





if __name__ == '__main__':
    cc_segment_length = 600             # CC segment length (in seconds)
    cc_segment_overlap = 0.5            # overlap of CC segments when creating subsamples (0.0 = no overlap)
    cc_segment_soc_bounds = (30,90)     # use only CC data between these state-of-charge bounds

