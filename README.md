# rapid-soh-estimation-from-short-pulses
This repository contains the Python scripts and processed data required to recreate the results and figures presented in the paper: "Rapid Estimation of Lithium-Ion Battery Capacity and Resistance from Short Duration Current Pulses" 

*Questions pertaining to the scripts and data provided in this repository can be directed to Ben Nowacki (benjamin.nowacki@uconn.edu)*


##  Paper Overview & Abstract

**Rapid Estimation of Lithium-Ion Battery Capacity and Resistance from Short Duration Current Pulses**

Ben Nowacki $^{1*}$, Jayanth Ramamurthy $^{2*}$, Adam Thelen $^{2}$, Chad Tischer $^{3}$, Cary L. Pint $^{2}$, Chao Hu $^{1,\dagger}$

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
        |- 01_some_notebook.ipynb
    |- processed_data/
        |- ...
        |- README.md
    |- rapid-soh-estimation/
        |- rapid-soh-estimation/
            |- __init__.py
            |- config.py
            |- common_methods.py
        |- setup.py
    |- LICENSE
    |- README.md
```

### `figures/`

### `notebooks/`

### `processed_data/`

### `rapid-soh-estimation/`


