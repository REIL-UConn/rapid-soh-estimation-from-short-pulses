
from config import *


def interp_time_series(ts:np.ndarray, ys:np.ndarray, n_points:int):
    '''
    Interpolates all y arrays to n_points based on a shared time array

    ts: Array,
    \tAn array of time values corresponding to every entry in ys
    ys : Array(s),
    \tA single array (or several stacked arrays) of values corresponding to ts
    n_points: int,
    \tThe output length of ts and ys

    Returns: A tuple of interpolated time values and corresponding y values (ts_interp, ys_interp).
    *Note that ys_interp will have the same shape as ys*
    '''


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

def clean_time_series_features(ts, ys):
    '''
    Removes duplicate timestamps and corresponding entries in ys
    ts: array of time values corresponding to every entry in ys
    ys: a single array (or several stacked arrays) of values corresponding to ts
    return:  (ts_clean, ys_clean)
    Note that ys_clean will have the same data type as ys
    '''

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


if __name__ == '__main__':
    print('common_methods.py')