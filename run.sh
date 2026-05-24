model_name='gpt2'
dataset_name='mydataset'

python get_predictions.py \
    --dataset_name $dataset_name \
    --model_name $model_name \
    --mode STR

python main.py \
    --dataset_name $dataset_name \
    --model_name $model_name \
    --metric causal

python score_utils/compute_scores.py \
    --dataset_name $dataset_name \
    --model_name $model_name \
    --metric causal \
    --causal_type STR

python score_utils/plot.py \
    --dataset_name $dataset_name \
    --model_name $model_name \
    --mode STR

python inspect_results.py \
    --model_name $model_name \
    --dataset_name $dataset_name