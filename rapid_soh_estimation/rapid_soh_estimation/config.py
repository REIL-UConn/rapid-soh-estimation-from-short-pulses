
#region: import statements
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import keras
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import tensorflow as tf
import optuna
from itertools import product

from datetime import datetime
from copy import copy, deepcopy

import pickle
from pathlib import Path
import warnings
import os, sys
import re
from enum import Enum
#endregion

cwd = os.path.abspath(__file__)
dir_repo_main = Path(str(cwd)[:str(cwd).rindex('rapid-soh-estimation-from-short-pulses') + len('rapid-soh-estimation-from-short-pulses')])
assert dir_repo_main.is_dir()


# =================================================================================
#   USER-SPECIFIC PATH DEFINITONS
# =================================================================================
dir_preprocessed_data = Path("/Users/bnowacki/Library/CloudStorage/Dropbox/Datasets to Publish/UConn-ISU-ILCC LFP battery aging")
dir_NMC_data = Path("/Users/bnowacki/Library/CloudStorage/Dropbox/Battery Repurposing Data 2/NMC Dataset/RWTH-2021-04545_818642")
# TODO: ABOVE LINES ARE USER SPECIFIC (ALSO CHANGE NOTEBOOKS TO TEMPLATE)




# =================================================================================
#   GLOBAL PATH DEFINITONS
# =================================================================================
dir_figures = dir_repo_main.joinpath("figures")
dir_notebooks = dir_repo_main.joinpath("notebooks")
dir_processed_data = dir_repo_main.joinpath("processed_data")
dir_spreadsheets = dir_repo_main.joinpath("spreadsheets")
dir_results = dir_repo_main.joinpath("results")

# =================================================================================
#   GLOBAL VARIABLES
# =================================================================================
path_test_tracker = dir_spreadsheets.joinpath("Cell_Test_Tracker.xlsx")
df_test_tracker = pd.read_excel(path_test_tracker, sheet_name=0, engine='openpyxl')

path_v_vs_soc_1c_chg = dir_spreadsheets.joinpath("V vs SOC 1C Charge.csv")
df_v_vs_soc_1c_chg = pd.read_csv(path_v_vs_soc_1c_chg)

if __name__ == '__main__':
    print('config.py()')