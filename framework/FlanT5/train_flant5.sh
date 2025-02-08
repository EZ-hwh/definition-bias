HOME_PATH=/data2/huangwenhao/hf_model

CUDA_LAUNCH_BLOCKING=0 deepspeed --include=localhost:0,1,2,3,6,7 \
    train_flant5.py \
    --max_epoches 5 \
    --data_path train_1214.jsonl \
    --model_path $HOME_PATH/flan-t5-xxl \
    --ds_config_path ds_config_flan_t5_xxl.json \
    --save_dir ckp/flan_t5_weighted \
    --save_name ckp \
    --save_steps 1000000 \
    --max_length 512
