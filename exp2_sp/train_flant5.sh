DATA_PATH="/mnt/data122/datasets/Information\ Extraction/academic_dataset/IE_INSTRUCTIONS"
HOME_PATH=/data2/huangwenhao

deepspeed --include=localhost:6,7 \
    train_flant5.py \
    --max_epoches 5 \
    --data_path /mnt/data122/datasets/Information\ Extraction/academic_dataset/IE_INSTRUCTIONS \
    --train_config train.yaml \
    --model_path $HOME_PATH/flan-t5-xxl \
    --ds_config_path configs/ds_config_flan_t5_xxl.json \
    --save_dir ckp/flant5_source_nickname \
    --save_name ckp \
    --save_steps 5000 \
    --max_length 512
