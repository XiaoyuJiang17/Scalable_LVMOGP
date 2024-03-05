#!/bin/bash

for seed in 3 4;
do  
    ## OurModel

    # synthetic
    # python ./run_experiments/run_OurModel.py --config_name synthetic/Scale_RBF/lvmogp_50_outputs_unfix --random_seed $seed
    # python ./run_experiments/run_OurModel.py --config_name synthetic/Scale_RBF/lvmogp_100_outputs_unfix --random_seed $seed
    # python ./run_experiments/run_OurModel.py --config_name synthetic/Scale_RBF/lvmogp_500_outputs_unfix --random_seed $seed
    # python ./run_experiments/run_OurModel.py --config_name synthetic/Scale_RBF/lvmogp_1000_outputs_unfix --random_seed $seed
    # python ./run_experiments/run_OurModel.py --config_name synthetic/Scale_RBF/lvmogp_5000_outputs_unfix --random_seed $seed

    # mocap
    # python ./run_experiments/run_OurModel.py --config_name mocap/Scale_RBF/lvmogp_08_02_unfix --random_seed $seed

    # exchange
    # python ./run_experiments/run_OurModel.py --config_name exchange/Scale_RBF/lvmogp_unfix --random_seed $seed


    ## IndepSVGP

    # synthetic
    python ./run_experiments/run_IndepSVGP.py --config_name synthetic_IGP/Scale_RBF/IndepSVGP_50_outputs_unfix --random_seed $seed
    python ./run_experiments/run_IndepSVGP.py --config_name synthetic_IGP/Scale_RBF/IndepSVGP_100_outputs_unfix --random_seed $seed
    python ./run_experiments/run_IndepSVGP.py --config_name synthetic_IGP/Scale_RBF/IndepSVGP_500_outputs_unfix --random_seed $seed
    python ./run_experiments/run_IndepSVGP.py --config_name synthetic_IGP/Scale_RBF/IndepSVGP_1000_outputs_unfix --random_seed $seed
    python ./run_experiments/run_IndepSVGP.py --config_name synthetic_IGP/Scale_RBF/IndepSVGP_5000_outputs_unfix --random_seed $seed

    # mocap
    # python ./run_experiments/run_IndepSVGP.py --config_name mocap_IGP/Scale_RBF/IndepSVGP_08_02_unfix --random_seed $seed
done