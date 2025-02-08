#dataset=conll04
MODEL_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/models
DATA_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/huangwenhao/data/IE_INSTRUCTIONS/RE

for dataset in 'GIDS'
do
    echo $dataset
    CUDA_VISIBLE_DEVICES=0 python ext_pt_eng.py \
    --model_path=$MODEL_PATH/bert-large-cased \
    --do_test=True \
    --data_path=/$DATA_PATH/$dataset \
    --dname=$dataset \
    --save_path=./weight/$dataset \
    --maxlen=512
done

# for dataset in 'HacRED' 'SKE2019' 'DuIE2.0'
# do
#     echo $dataset
#     CUDA_VISIBLE_DEVICES=7 python ext_pt_eng.py \
#     --model_path=$MODEL_PATH/chinese-roberta-wwm-ext-large \
#     --do_test=True \
#     --data_path=/$DATA_PATH/$dataset \
#     --dname=$dataset \
#     --save_path=./weight/$dataset \
#     --maxlen=512
# done