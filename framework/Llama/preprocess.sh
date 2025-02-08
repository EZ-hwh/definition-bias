HOME_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/models

#python ochat/data/generate_dataset.py \
python -m ochat.data.generate_dataset \
    --model-type openchat_v3.2 \
    --model-path $HOME_PATH/llama-2-13b-hf \
    --in-files 'dataset/original_1214.jsonl' \
    --out-prefix 'processed_dataset/original_1214'
    #--eval-ratio 0