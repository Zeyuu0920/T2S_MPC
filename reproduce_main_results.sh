#!/bin/bash

set -e

echo "Running stabilization experiments for table 1"
python scripts/run_stabilization_Nominal_MPC.py --wind linear --slope 0.0001 --seed 42  --noise_std 0.0005
python scripts/run_stabilization_Neural_MPC.py --wind linear --slope 0.0001 --seed 42  --noise_std 0.0005
python scripts/run_stabilization_T2S_MPC.py --wind linear --slope 0.0001 --seed 42  --noise_std 0.0003
python scripts/run_stabilization_T2S_wo_time_emd.py --wind linear --slope 0.0001 --seed 42  --noise_std 0.0005
python scripts/run_stabilization_T2S_wo_two_scales.py --wind linear --slope 0.0001 --seed 42  --noise_std 0.0005
python scripts/run_stabilization_Nominal_MPC.py --wind periodic --A 0.003 --period 2 --seed 42  --noise_std 0.0005
python scripts/run_stabilization_Neural_MPC.py --wind periodic --A 0.003 --period 2 --seed 42  --noise_std 0.0005
python scripts/run_stabilization_T2S_MPC.py --wind periodic --A 0.003 --period 2 --seed 42  --noise_std 0.0005
python scripts/run_stabilization_T2S_wo_time_emd.py --wind periodic --A 0.003 --period 2 --seed 42  --noise_std 0.0005
python scripts/run_stabilization_T2S_wo_two_scales.py --wind periodic --A 0.003 --period 2 --seed 42  --noise_std 0.0005

echo "Running tracking experiments for table 2"
python scripts/run_tracking_circle_Nominal_MPC.py --wind linear --slope 0.0001 --seed 42  --noise_std 0.0005
python scripts/run_tracking_circle_Neural_MPC.py --wind linear --slope 0.0001 --seed 42  --noise_std 0.0005
python scripts/run_tracking_circle_T2S_MPC.py --wind linear --slope 0.0001 --seed 42  --noise_std 0.0005
python scripts/run_tracking_circle_Nominal_MPC.py --wind periodic --A 0.003 --period 2 --seed 42  --noise_std 0.0005
python scripts/run_tracking_circle_Neural_MPC.py --wind periodic --A 0.003 --period 2 --seed 42  --noise_std 0.0005
python scripts/run_tracking_circle_T2S_MPC.py --wind periodic --A 0.003 --period 2 --seed 42  --noise_std 0.0005
python scripts/run_tracking_figure8_Nominal_MPC.py --wind linear --slope 0.0001 --seed 42  --noise_std 0.0005
python scripts/run_tracking_figure8_Neural_MPC.py --wind linear --slope 0.0001 --seed 42 --noise_std 0.0005
python scripts/run_tracking_figure8_T2S_MPC.py --wind linear --slope 0.0001 --seed 42 --noise_std 0.0005
python scripts/run_tracking_figure8_Nominal_MPC.py --wind periodic --A 0.003 --period 2 --seed 42 --noise_std 0.0005
python scripts/run_tracking_figure8_Neural_MPC.py --wind periodic --A 0.003 --period 2 --seed 42 --noise_std 0.0005
python scripts/run_tracking_figure8_T2S_MPC.py --wind periodic --A 0.003 --period 2 --seed 42 --noise_std 0.0005

echo "Collect results and compute for Fig.4"
python scripts/plot_stabilization_error_linear.py

echo "Collect results and compute for Fig.5"
python scripts/plot_stabilization_error_periodic.py

echo "Finished."