declare -a mlmmodels=("bert-base-uncased" "bert-large-uncased" "roberta-base" "roberta-large" "distilbert-base-uncased")
declare -a incrementalmodels=("openai-gpt" "gpt2" "gpt2-medium" "distilgpt2")

echo "Running POPLM experiments on Incremental Language Models"

for model in ${incrementalmodels[@]}    
do
    echo "Running experiments for ${model} on dataset: CORE_dative_1500sampled.csv!"
    python ../python/struct_priming_pop_lm.py --model ${model} --device cuda --lmtype incremental --batchsize 10 --committee_size 50 --dataset_name "core-da" --dataset_path ../data/PrimeLM_sampled/CORE_dative_1500sampled.csv
    echo "Running experiments for ${model} on dataset: CORE_transitive_1500sampled.csv!"
    python ../python/struct_priming_pop_lm.py --model ${model} --device cuda --lmtype incremental --batchsize 10 --committee_size 50 --dataset_name "core-tr" --dataset_path ../data/PrimeLM_sampled/CORE_transitive_1500sampled.csv
done

echo "Running POPLM experiments on Masked Language Models"

for model in ${mlmmodels[@]}
do
    echo "Running experiments for ${model} on dataset: CORE_dative_1500sampled.csv!"
    python ../python/struct_priming_pop_lm.py --model ${model} --device cuda --batchsize 10 --committee_size 50 --dataset_name "core-da" --dataset_path ../data/PrimeLM_sampled/CORE_dative_1500sampled.csv
    echo "Running experiments for ${model} on dataset: CORE_transitive_1500sampled.csv!"
    python ../python/struct_priming_pop_lm.py --model ${model} --device cuda --batchsize 10 --committee_size 50 --dataset_name "core-tr" --dataset_path ../data/PrimeLM_sampled/CORE_transitive_1500sampled.csv
done