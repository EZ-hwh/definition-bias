HOME_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/models
#NUM_GPUS=7
BASE_MODEL=main_1214
# for dataset in 'Ontonotes' 'New-York-Times-RE' 'NYT11' 'WikiNeural' 'PolyglotNER'  
# do
#     deepspeed --include=localhost:0,1,2,3,4,5,6,7 train_deepspeed_lora.py \
#         --max_epoches 10 \
#         --save_dir "lora_ckp/$BASE_MODEL" \
#         --save_name "$dataset" \
#         --model_path "ckp/$BASE_MODEL/ep_4" \
#         --dataset_type GPT2Dataset_onlyres \
#         --data_path "dataset/single/$dataset.jsonl" \
#         --max_length 1024 \
#         --ds_config_path=ds_config_llama_lora_13B_1024.json
# done

for dataset in 'Ontonotes' 'New-York-Times-RE' 'NYT11' 'WikiNeural' 'PolyglotNER'  
do
    deepspeed --include=localhost:0,1,2,3,4,5,6,7 train_deepspeed_lora.py \
        --max_epoches 10 \
        --save_dir "lora_ckp/No_tuning" \
        --save_name "$dataset" \
        --model_path "$HOME_PATH/llama-2-13b-hf" \
        --dataset_type GPT2Dataset_onlyres \
        --data_path "dataset/single/$dataset.jsonl" \
        --max_length 1024 \
        --ds_config_path=ds_config_llama_lora_13B_1024.json
done