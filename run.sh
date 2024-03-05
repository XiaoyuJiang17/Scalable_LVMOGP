#!/bin/bash

# python ./run_experiments/run_OurModel.py --config_name exchange/Scale_RBF/lvmogp_unfix --random_seed 1
# python ./run_experiments/run_OurModel.py --config_name egg/Scale_RBF/lvmogp_unfix --random_seed 1

python ./run_experiments/run_OurModel.py --config_name mocap/Scale_RBF/lvmogp_64_08 --random_seed 1
# python ./run_experiments/run_IndepSVGP.py --config_name mocap_IGP/Scale_RBF/IndepSVGP_64_08 --random_seed 1