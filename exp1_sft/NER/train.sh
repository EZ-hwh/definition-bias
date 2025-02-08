MODEL_PATH=/data2/huangwenhao/hf_model
DATA_PATH="/mnt/data122/datasets/Information Extraction/academic_dataset/IE_INSTRUCTIONS/NER"

for dataset in 'BioRED'
do
    CUDA_VISIBLE_DEVICES=2 python extraction.py \
    --do_train=True \
    --plm=$MODEL_PATH/bert-large-cased \
    --train_dp="$DATA_PATH/$dataset" \
    --weight_file=gp_$dataset.pt
done