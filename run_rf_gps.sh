#!/bin/bash
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly3_a4.json --datasets_file config/active_datasets2.json --use_gpu --zero_center
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly7_a4.json --datasets_file config/active_datasets2.json --use_gpu --zero_center
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly10_a4.json --datasets_file config/active_datasets2.json --use_gpu --zero_center

python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly3_a4.json --datasets_file config/active_datasets2.json --use_gpu
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly7_a4.json --datasets_file config/active_datasets2.json --use_gpu
python run_rf_gp_experiments.py --rf_parameter_file config/rf_parameters/poly10_a4.json --datasets_file config/active_datasets2.json --use_gpu
