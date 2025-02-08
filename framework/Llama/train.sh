HOME_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/models
NUM_GPUS=8

#deepspeed --include=localhost:0,1,2,3,4,5,6,7 train_deepspeed.py \
# deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=12345 train_deepspeed.py \
#           --model_path $HOME_PATH/llama-2-13b-hf \
#           --data_prefix processed_dataset/train_1214 \
#           --save_path ckp/main_v3 \
#           --batch_max_len 1024 \
#           --epochs 5 \
#           --save_every 1 \
#           --deepspeed \
#           --deepspeed_config deepspeed_config.json

deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=12345 train_deepspeed.py \
          --model_path $HOME_PATH/llama-2-13b-hf \
          --data_prefix processed_dataset/original_1214 \
          --save_path ckp/original_1214 \
          --batch_max_len 1024 \
          --epochs 5 \
          --save_every 1 \
          --deepspeed \
          --deepspeed_config deepspeed_config.json