#!/bin/bash
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly3_ctr.json --datasets_file config/active_datasets2.json --use_gpu
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly7_ctr.json --datasets_file config/active_datasets2.json --use_gpu
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly10_ctr.json --datasets_file config/active_datasets2.json --use_gpu
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly20_ctr.json --datasets_file config/active_datasets2.json --use_gpu