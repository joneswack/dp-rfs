#!/bin/bash
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly3.json --datasets_file config/active_datasets2.json --use_gpu
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly7.json --datasets_file config/active_datasets2.json --use_gpu
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly10.json --datasets_file config/active_datasets2.json --use_gpu
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly20.json --datasets_file config/active_datasets2.json --use_gpu