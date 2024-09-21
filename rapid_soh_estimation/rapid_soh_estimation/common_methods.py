
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
	
def load_processed_data(file_paths) -> pd.DataFrame:
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



if __name__ == '__main__':
    print('common_methods.py')