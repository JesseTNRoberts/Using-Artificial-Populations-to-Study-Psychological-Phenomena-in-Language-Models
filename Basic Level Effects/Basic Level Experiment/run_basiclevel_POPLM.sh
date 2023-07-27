# !/bin/bash

declare -a mlmmodels=("bert-base-uncased")
declare -a incrementalmodels=("gpt2")

echo "Running POPLM experiments on Masked Language Models"

for model in ${mlmmodels[@]}
do
    echo "Running experiments for ${model}!"
    python3 taxonomic_verification_pop_lm.py --model ${model} --device cpu --batchsize 162 --dataset basic_level_experiment.csv
done

echo "Running POPLM experiments on Incremental Language Models"

for model in ${incrementalmodels[@]}
do
    echo "Running experiments for ${model}!"
    python3 taxonomic_verification_pop_lm.py --model ${model} --device cpu --lmtype incremental --batchsize 162 --dataset basic_level_experiment.csv
done
