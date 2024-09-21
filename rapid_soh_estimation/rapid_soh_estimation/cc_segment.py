
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





if __name__ == '__main__':
    cc_segment_length = 600             # CC segment length (in seconds)
    cc_segment_overlap = 0.5            # overlap of CC segments when creating subsamples (0.0 = no overlap)
    cc_segment_soc_bounds = (30,90)     # use only CC data between these state-of-charge bounds

