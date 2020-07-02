export CUDA_NO=0
export DATA_DIR=../data/pdtb2_xval/fold_2
export TASK_NAME=PDTB2_LEVEL2

python ../src/pytorch-transformers/examples/run_pdtb.py \
    --model_type bert \
    --task_name $TASK_NAME \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --evaluate_during_training \
    --do_eval \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size 32 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 5e-6 \
    --num_train_epochs 10.0 \
    --output_dir output/fold_2  \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 1 \
    --validation_metric acc \
    --n_gpu 1 \
    --cuda_no ${CUDA_NO} \
    --deterministic
