# rapid-soh-estimation-from-short-pulses
This repository contains the Python scripts and processed data required to recreate the results and figures presented in the paper: "Rapid Estimation of Lithium-Ion Battery Capacity and Resistance from Short Duration Current Pulses" ([https://doi.org/10.1016/j.jpowsour.2024.235813](https://doi.org/10.1016/j.jpowsour.2024.235813))

*Questions pertaining to the scripts and data provided in this repository can be directed to Ben Nowacki (benjamin.nowacki@uconn.edu)*



##  Paper Details & Abstract

### Rapid Estimation of Lithium-Ion Battery Capacity and Resistance from Short Duration Current Pulses

Benjamin Nowacki $^{1*}$, Jayanth Ramamurthy $^{2*}$, Adam Thelen $^{2}$, Chad Tischer $^{3}$, Cary L. Pint $^{2}$, Chao Hu $^{1,\dagger}$

&nbsp;&nbsp;&nbsp;&nbsp; $^{1}$ Department of Mechanical Engineering, University of Connecticut, Storrs, CT 06269

&nbsp;&nbsp;&nbsp;&nbsp; $^{2}$ Department of Mechanical Engineering, Iowa State University, Ames, IA 50011

&nbsp;&nbsp;&nbsp;&nbsp; $^{3}$ Department of Engineering Technology, Iowa Lakes Community College, Estherville, IA 51334

&nbsp;&nbsp;&nbsp;&nbsp; $^{*}$ Indicates authors contributed equally

&nbsp;&nbsp;&nbsp;&nbsp; $^{\dagger}$ Indicates corresponding author. Email: chao.hu@uconn.edu


**Abstract**:
Rapid onboard diagnosis of battery state of health enables the use of real-time control strategies that can improve product safety and maximize battery lifetime. However, onboard prediction of battery state-of-health is challenging due to the limitations imposed on diagnostic tests so as not to negatively affect the user experience and impede normal operation. To this end, we demonstrate a lightweight machine learning model capable of predicting a lithium-ion battery's discharge capacity and internal resistance at various states of charge using only the raw voltage-capacity time-series data recorded during short-duration ($100$ seconds) current pulses. Tested on two battery aging datasets, one publicly available and the other newly collected for this work, we find that the best models can accurately predict cell discharge capacity with an average mean-absolute-percent-error of 1.66\%. Additionally, we quantize and embed the machine learning model onto a microcontroller and show comparable accuracy to the computer-based model, further demonstrating the practicality of on-board rapid capacity and resistance estimation. 


## Repository Structure

```
|- rapid-soh-estimation-from-short-pulses/
    |- figures/
        |- final/
        |- raw/
    |- notebooks/ 
    |- processed_data/
        |- LFP/
        |- NMC/
    |- rapid-soh-estimation/
        |- rapid-soh-estimation/
            |- __init__.py
            |- config.py
            |- common_methods.py
            |- postprocesing.py
            |- cc_segment.py
            |- slowpulse.py
            |- quantization.py
            |- plotting.py
        |- setup.py
    |- environment.yml
    |- LICENSE
    |- README.md
```

### `figures/`

This directory contains all figures generated and used in the published paper. The subfolder 'final' contains the eight figures provided in the main content of the published paper. The 'raw' subfolder contains many other figures output by the subsequently defined scripts. 



### `notebooks/`

The notebooks directory contains several Jupyter notebooks that were used to conduct the majority of the analysis. Common methods/functions are refactored to Python scripts contained within the `rapid-soh-estimation/rapid-soh-estimation/` directory. 

* `../feature_importance.ipynb`: Provides implementation details on how the pulse feature importance (Figure 8) was evaluated. It also plots the train and test accuracy of the Ridge, Lasso, ElasticNet, RandomForest, and MLP models discussed in the feature importance section of the paper.
* `../input_features.ipynb`: Provides visual description of the pulse voltage and CC voltage segment input features. The aging trajectories of the LFP/Gr cells is also shown.
* `../model_comparisons.ipynb`: Shows the implementation details on the model type comparisons, input type comparisons, and pulse SOC comparisons.
* `../model_optimization.ipynb`: Provides the implementation details on how models were optimized for each cell chemsitry and pulse type using the Optuna framework. Model evaluation on out-of-distribution pulses and the optimal model size are also shown.
* `../soc_drift.ipynb`: Shows how SOC drift is calculate and visualized for the LFP/Gr cells. 



### `processed_data/`

Note that due to the size of all raw data, only processed data is retained in this Git repository. The raw data collected on the 64 LFP/Gr cells can be downloaded at: 

$\textcolor{red}{\text{LINK TO LFP/Gr DATASET WILL BE PUBLISHED SOON}}$

Please see the 'README.md' file contained in the above linked data for an overview of how the raw data files are structured.

The 48 NMC/Gr dataset is published by Aachen University under the title: *Time-series cyclic aging data on 48 commercial NMC/graphite Sanyo/Panasonic UR18650E cylindrical cells* and can be downloaded at: [https://publications.rwth-aachen.de/record/818642](https://publications.rwth-aachen.de/record/818642). 

Note that the LFP data (`processed_data/LFP/`) is subdivided into processed data for the different feature selections: data for the constant-current model is stored in `processed_data/LFP/cc`, and data for the voltage pulse models in stored in `processed_data/LFP/slowpulse`. The processed NMC data (`processed_data/NMC/`) is organized with one processed file for each cell ID.



### `rapid-soh-estimation/rapid-soh-estimation/`

This folder contains scripts pertaining to unique aspects of data processing and analysis.

* `../config.py`
  - Contains the set of imports used throughout this repository as well as global variables and path definitions 
  - **IMPORTANT**: You must set the local paths to the downloaded data for the LFP and NMC datasets: `dir_preprocessed_data` and `dir_NMC_data`, respectively.
* `../common_methods.py`
  - Contains a set of functions commonly used within the subsequent Python scripts
* `../postprocessing.py`
  - Contains all code pertaining to processing the downloaded data for the LFP/Gr dataset. 
  - Running the script will process the downloaded data located at the path defined in `config.py`. Note that this *won't* overwrite previously processed data; it will simply save a new copy.  
* `../postprocessing_NMC.py`
  - Contains all code pertaining to processing the downloaded data for the NMC/Gr dataset. 
  - Running the script will process the downloaded raw data located at the path defined in `config.py`. Note that this *will* overwrite previously processed data. 
* `../slowpulse.py`
  - Contains the functions used to create a sequential neural network for SOH estimation using the voltage response of the short-duration current pulse as input
  - Running the script will create a new model for the defined parameters (chemistry, pulse type, spulse state of charge, etc)
  - See the `if __name__ == '__main__':` block of code for the full implementation details
* `../cc_segment.py`
  - Contains the functions used to create a sequential neural network for SOH estimation using a segment of the constant-current charge voltage as input
  - Running the script will create a new model for the defined parameters (chemistry, voltage, segment length, etc)
  - See the `if __name__ == '__main__':` block of code for the full implementation details
* `../quantization.py`
  - Contains the functions used to quantize a TensorFlow model for use on a microcontroller
  - Running the script will perform quantization on the most recent model with the specified parameters (pulse type, pulse soc, quanitization level, etc)
  - See the `if __name__ == '__main__':` block of code for the full implementation details
* `../plotting.py`
  - Serves to consolidate all plotting functions to a single location.
  - Running the script itself does nothing. Methods must be called externally



### `results/`

This directory contains saved results from several analsyses performed. Files are mostly saved in the binary Pickle format. See `notebooks` and `rapid-soh-estimation/rapid-soh-estimation/` for details on how to properly load and use these results.



### `spreadsheets/`

This directory contains two spreadsheets: 'Cell_Test_Tracker.xlsx' and 'V vs SOC 1C Charge.csv'. The former provides a simple mapping between cell ID, tester details, and cycling conditions. The latter is an interpolated lookup table of cell terminal voltage and state-of-charge (defined with coulomb counting) on an LFP/Gr cell during a 1C CC-CV charge.


