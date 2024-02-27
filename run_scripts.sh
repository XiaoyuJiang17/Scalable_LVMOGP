#!/bin/bash

for seed in {1..5}
do
    python ./run_experiments/run_OurModel.py --config_name mocap/Scale_RBF/lvmogp_08_02_unfix --random_seed $seed
done