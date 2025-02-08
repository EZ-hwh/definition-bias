HOME_PATH=/data2/huangwenhao/hf_model
NUM_GPUS=7
BASE_MODEL=flan_t5_weighted
# deepspeed --num_gpus=$NUM_GPUS train_deepspeed_lora.py \
#           --model_path ckp/original/ep_4 \
#           --data_prefix 'processed_dataset/single/ACE 2004' \
#           --save_path 'lora_ckp/original/ACE 2004' \
#           --batch_max_len 1024 \
#           --epochs 10 \
#           --save_every 1 \
#           --deepspeed \
#           --deepspeed_config deepspeed_config.json

# for dataset in 'ACE 2004' 'ACE 2005' 'CoNLL 2003' 'conll04' 'GIDS' 'TweetNER7' 'WikiANN en' 'WikiKBP'
# do
#     deepspeed --include=localhost:2,3,4,5,6,7 train_flant5.py \
#         --max_epoches 30 \
#         --save_dir "lora_ckp/$BASE_MODEL" \
#         --save_name "$dataset" \
#         --use_lora \
#         --model_path ckp/$BASE_MODEL/ckp_epoch4 \
#         --dataset_type BertDataset_onlyres \
#         --data_path "../dataset/single/$dataset.jsonl" \
#         --max_length 512 \
#         --ds_config_path=ds_config_flan_t5_lora.json
# done

# for dataset in 'Ontonotes' 'WikiNeural' 'New-York-Times-RE' 'NYT11' 'PolyglotNER'
# do
#     deepspeed --include=localhost:2,3,4,5,6,7 train_flant5.py \
#         --max_epoches 10 \
#         --save_dir "lora_ckp/$BASE_MODEL" \
#         --save_name "$dataset" \
#         --use_lora \
#         --model_path ckp/$BASE_MODEL/ckp_epoch4 \
#         --dataset_type BertDataset_onlyres \
#         --data_path "../dataset/single/$dataset.jsonl" \
#         --max_length 512 \
#         --ds_config_path=ds_config_flan_t5_lora.json
# done

#for dataset in 'ACE 2004' 'ACE 2005' 'CoNLL 2003' 'conll04' 'GIDS' 'TweetNER7' 'WikiANN en' 'WikiKBP'
# for dataset in 'WikiANN en' 'WikiKBP'
# do
#     deepspeed --include=localhost:1,3,4,5,6,7 train_flant5.py \
#         --max_epoches 30 \
#         --save_dir "lora_ckp/original" \
#         --save_name "$dataset" \
#         --use_lora \
#         --model_path /data2/huangwenhao/hf_model/flan-t5-xxl \
#         --dataset_type BertDataset_onlyres \
#         --data_path "../dataset/single/$dataset.jsonl" \
#         --max_length 512 \
#         --ds_config_path=ds_config_flan_t5_lora.json
# done

for dataset in 'Ontonotes' 'WikiNeural' 'New-York-Times-RE' 'NYT11' 'PolyglotNER'
do
    deepspeed --include=localhost:0,1,2,4 train_flant5.py \
        --max_epoches 10 \
        --save_dir "lora_ckp/original" \
        --save_name "$dataset" \
        --use_lora \
        --model_path /data2/huangwenhao/hf_model/flan-t5-xxl \
        --dataset_type BertDataset_onlyres \
        --data_path "../dataset/single/$dataset.jsonl" \
        --max_length 512 \
        --ds_config_path=ds_config_flan_t5_lora.json
done