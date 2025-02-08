#dataset=conll04
MODEL_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/models
DATA_PATH=/cpfs01/projects-HDD/cfff-0f987ac9dfb4_HDD/liangjiaqing/huangwenhao/data/IE_INSTRUCTIONS/RE

# for train_ds in 'WikiKBP' 'conll04' 'GIDS' 'New-York-Times-RE' 'NYT11'
# do
#     for test_ds in 'WikiKBP' 'conll04' 'GIDS' 'New-York-Times-RE' 'NYT11'
#     do
#         echo $dataset
#         CUDA_VISIBLE_DEVICES=1 python cross_test.py \
#         --model_path=$MODEL_PATH/bert-large-cased \
#         --train_dp=$DATA_PATH/$train_ds \
#         --test_dp=$DATA_PATH/$test_ds \
#         --test_file=$DATA_PATH/$test_ds/test.json \
#         --result_path=cross_result/no_filter \
#         --refer_weight_file=./weight/$test_ds \
#         --test_weight_file=./weight/$train_ds \
#         --maxlen=512
#     done
# done

for train_ds in 'WikiKBP' 'conll04' 'GIDS' 'New-York-Times-RE' 'NYT11'
do
    for test_ds in 'WikiKBP' 'conll04' 'GIDS' 'New-York-Times-RE' 'NYT11'
    do
        if [ "$train_ds" != "$test_ds" ] ; then
            echo "test $train_ds with $test_ds"
            CUDA_VISIBLE_DEVICES=3 python cross_test.py \
            --model_path=$MODEL_PATH/bert-large-cased \
            --train_dp=$DATA_PATH/$train_ds \
            --test_dp=$DATA_PATH/$test_ds \
            --test_file=../filter_dataset/$train_ds\_$test_ds.json \
            --result_path=cross_result/filter_0.8 \
            --refer_weight_file=./weight/$test_ds \
            --test_weight_file=./weight/$train_ds \
            --maxlen=512
        fi
    done
done

# for train_ds in 'HacRED' 'SKE2019' 'DuIE2.0'
# do
#     for test_ds in 'HacRED' 'SKE2019' 'DuIE2.0'
#     do
#         echo $dataset
#         CUDA_VISIBLE_DEVICES=7 python cross_test.py \
#         --model_path=$MODEL_PATH/chinese-roberta-wwm-ext-large \
#         --train_dp=/$DATA_PATH/$train_ds \
#         --test_dp=/$DATA_PATH/$test_ds \
#         --refer_weight_file=./weight/$test_ds \
#         --test_weight_file=./weight/$train_ds \
#         --maxlen=512
#     done
# done