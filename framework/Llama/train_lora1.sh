HOME_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/models
NUM_GPUS=7
BASE_MODEL=original_1214
# deepspeed --num_gpus=$NUM_GPUS train_deepspeed_lora.py \
#           --model_path ckp/original/ep_4 \
#           --data_prefix 'processed_dataset/single/ACE 2004' \
#           --save_path 'lora_ckp/original/ACE 2004' \
#           --batch_max_len 1024 \
#           --epochs 10 \
#           --save_every 1 \
#           --deepspeed \
#           --deepspeed_config deepspeed_config.json

for dataset in 'ACE 2004' 'ACE 2005' 'CoNLL 2003' 'conll04' 'GIDS' 'TweetNER7' 'WikiANN en' 'WikiKBP'
do
    deepspeed --include=localhost:1,2,3,4,5,6,7 train_deepspeed_lora.py \
        --max_epoches 30 \
        --save_dir "lora_ckp/$BASE_MODEL" \
        --save_name "$dataset" \
        --model_path ckp/$BASE_MODEL/ep_4 \
        --dataset_type BertDataset_onlyres \
        --data_path "dataset/single/$dataset.jsonl" \
        --max_length 512 \
        --ds_config_path=ds_config_llama_lora_13B_512.json
done
