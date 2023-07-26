#!/bin/bash
# MAKE batchsize the length of the file -- currently batching is not supported so it is required to use one batch

declare -a mlmmodels=("bert-large-uncased" "roberta-base" "roberta-large" "distilbert-base-uncased")
declare -a incrementalmodels=("openai-gpt" "gpt2" "gpt2-medium" "distilgpt2")

echo "Running recreation experiments on Masked Language Models"

for model in ${mlmmodels[@]}
do
    echo "Running experiments for ${model}!"
    python3 ../python/taxonomic_verification_recreate.py --model ${model} --device cpu --batchsize 565 --dataset ../data/rosch1975/rosch1975_alternate.csv
done

echo "Running recreation experiments on Incremental Language Models"

for model in ${incrementalmodels[@]}
do
    echo "Running experiments for ${model}!"
    python3 ../python/taxonomic_verification_recreate.py --model ${model} --device cpu --lmtype incremental --batchsize 565 --dataset ../data/rosch1975/rosch1975_alternate.csv
done


echo "Running POPLM experiments on Masked Language Models"

for model in ${mlmmodels[@]}
do
    echo "Running experiments for ${model}!"
    python3 ../python/taxonomic_verification_pop_lm.py --model ${model} --device cpu --batchsize 565 --dataset ../data/rosch1975/rosch1975_alternate.csv
done

echo "Running POPLM experiments on Incremental Language Models"

for model in ${incrementalmodels[@]}
do
    echo "Running experiments for ${model}!"
    python3 ../python/taxonomic_verification_pop_lm.py --model ${model} --device cpu --lmtype incremental --batchsize 565 --dataset ../data/rosch1975/rosch1975_alternate.csv
done
