# Introduction
This repository contains code related to parametrization of equivalent circuit models in task 5.4


# Contents
- `ici_parametrization.py`: Parametrizes multiple ECM models for a series of ICI tests.
- `gitt_parametrization.py`: Parametrizes an ECM model from a GITT test.
- `ecm_model.py`: Defines an equivalent circuit model (ECM) uing the extracted ECM parameters 
- `module_model.py`: Implements a module model using the ECM model.   

# Using the repository
- `gitt_ecm_parameters.csv` contains $OCV$, $R_0$, $R_1$ and $\tau_1$ as a functions of SOC and hysteresis direction. 
- Re-run `bol_parametrization.py` as a scripts to re-generate `gitt_ocv_soc.csv`, `gitt_rc_parameters.csv` and `gitt_ecm_parameters.csv`.  