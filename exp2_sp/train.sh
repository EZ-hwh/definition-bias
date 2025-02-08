DATA_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/huangwenhao/data/IE_INSTRUCTIONS
HOME_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/models

# deepspeed --include=localhost:1,2,3,4,5,6,7 \
#     train.py \
#     --max_epoches 5 \
#     --data_path $DATA_PATH \
#     --train_config train.yaml \
#     --model_path $HOME_PATH/llama-2-13b-hf \
#     --ds_config_path configs/ds_config_llama_ft_13B_1024.json \
#     --flashattn \
#     --save_dir ckp/llama13b_source \
#     --save_name ckp \
#     --save_steps 5000 \
#     --dataset_type GPT2Dataset_onlyres \
#     --max_length 1024

deepspeed --include=localhost:0,1,2,3,4,5,6,7 \
    train.py \
    --max_epoches 5 \
    --data_path $DATA_PATH \
    --train_config train.yaml \
    --model_path $HOME_PATH/llama-2-13b-hf \
    --ds_config_path configs/ds_config_llama_ft_13B_1024.json \
    --flashattn \
    --use_nickname \
    --save_dir ckp/llama13b_source_nickname \
    --save_name ckp \
    --save_steps 5000 \
    --dataset_type GPT2Dataset_onlyres \
    --max_length 1024