export TRAIN_FILE=data/train_sentences.txt
export TEST_FILE=data/test_sentences.txt
#export TRAIN_FILE=data/train/t5.txt
#export TEST_FILE=data/test/t5.txt
#export TRAIN_FILE=data/train_subset_0.txt
#export TEST_FILE=data/test_subset_0.txt

pipenv run python run_language_modeling.py \
    --output_dir ./models/KariBERTa-small \
    --tokenizer_name models/KariBERTa \
    --model_type kariberta \
    --block_size 500 \
    --mlm \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 50 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_gpu_train_batch_size 16 \
    --evaluate_during_training \
    --seed 42 \
    --train_data_file $TRAIN_FILE \
    --eval_data_file $TEST_FILE

#    --fp16 \

