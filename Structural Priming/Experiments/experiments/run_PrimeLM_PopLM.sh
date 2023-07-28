declare -a mlmmodels=("bert-base-uncased" "bert-large-uncased" "roberta-base" "roberta-large" "distilbert-base-uncased")
declare -a incrementalmodels=("openai-gpt" "gpt2" "gpt2-medium" "distilgpt2")

echo "Running POPLM experiments on Masked Language Models"

for model in ${mlmmodels[@]}
do
    echo "Running experiments for ${model}!"
    python ../python/struct_priming_pop_lm.py --model ${model} --device cuda --batchsize 50 --num_batches 10 --dataset ../data/PrimeLM/CORE_transitive_15000sampled_10-1.csv
    python ../python/struct_priming_pop_lm.py --model ${model} --device cuda --batchsize 50 --num_batches 10 --dataset ../data/PrimeLM/CORE_dative_15000sampled_10-1.csv
done

echo "Running POPLM experiments on Incremental Language Models"

for model in ${incrementalmodels[@]}
do
    echo "Running experiments for ${model}!"
    python ../python/struct_priming_pop_lm.py --model ${model} --device cuda --lmtype incremental --batchsize 50 --num_batches 10 --dataset ../data/PrimeLM/CORE_transitive_15000sampled_10-1.csv
    python ../python/struct_priming_pop_lm.py --model ${model} --device cuda --lmtype incremental --batchsize 50 --num_batches 10 --dataset ../data/PrimeLM/CORE_dative_15000sampled_10-1.csv
done