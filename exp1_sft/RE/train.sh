#dataset=conll04
MODEL_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/models
DATA_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/huangwenhao/data/IE_INSTRUCTIONS/RE
#files=$(ls $DATA_PATH)
#for dataset in $files

for dataset in 'conll04'
do
    echo $dataset
    CUDA_VISIBLE_DEVICES=1 python ext_pt_eng.py \
    --model_path=$MODEL_PATH/bert-large-cased \
    --do_train=True \
    --data_path=/$DATA_PATH/$dataset \
    --dname=$dataset \
    --save_path=./weight/$dataset \
    --maxlen=512
done


# for dataset in 'DuIE1.0'
# do
#     echo $dataset
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ext_pt_eng.py \
#     --model_path=$MODEL_PATH/chinese-roberta-wwm-ext-large \
#     --do_train=True \
#     --data_path=$DATA_PATH/$dataset \
#     --dname=$dataset \
#     --save_path=./weight/$dataset \
#     --maxlen=512
# done