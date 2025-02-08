MODEL_PATH=/data2/huangwenhao/hf_model
DATA_PATH="/mnt/data122/datasets/Information Extraction/academic_dataset/IE_INSTRUCTIONS/NER"

# for train_dataset in 'ACE 2004' 'ACE 2005' 'CoNLL 2003' 'Ontonotes' 'WikiANN en' 'TweetNER7'  'PolyglotNER'
# do
#     for test_dataset in 'ACE 2004' 'ACE 2005' 'CoNLL 2003' 'Ontonotes' 'WikiANN en' 'TweetNER7'  'PolyglotNER'
#     do
#         CUDA_VISIBLE_DEVICES=1 python extraction.py \
#         --plm=$MODEL_PATH/bert-large-cased \
#         --do_test=True \
#         --result_path=cross/no_filter \
#         --test_file=$DATA_PATH/$test_dataset/test.json \
#         --weight_file="gp_$train_dataset.pt" \ 
#         --train_dp="$DATA_PATH/$train_dataset" \
#         --test_dp="$DATA_PATH/$test_dataset"
#     done
# done

for train_ds in 'bc4chemd' 'bc5cdr' 'E3C' 'ncbi' 'BioRED' 'bc2gm'
do
    for test_ds in 'bc4chemd' 'bc5cdr' 'E3C' 'ncbi' 'BioRED' 'bc2gm'
    do
        if [ "$train_ds" != "$test_ds" ] ; then
            echo "test $train_ds with $test_ds"
            CUDA_VISIBLE_DEVICES=4 python extraction.py \
            --plm=$MODEL_PATH/bert-large-cased \
            --do_test=True \
            --result_path=cross/biochem \
            --weight_file="gp_$train_ds.pt" \
            --train_dp="$DATA_PATH/$train_ds" \
            --test_dp="$DATA_PATH/$test_ds" \
            --test_file="$DATA_PATH/$test_ds/test.json"
            #--test_file="../filter_dataset/$train_ds"_"$test_ds.json" \
        fi
    done
done