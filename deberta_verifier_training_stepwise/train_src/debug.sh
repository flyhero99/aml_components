export CUDA_VISIBLE_DEVICES=0
export MAX_LENGTH=512
export BERT_MODEL=microsoft/deberta-v3-large
export OUTPUT_DIR=../../debug
export BATCH_SIZE=
export NUM_EPOCHS=16
export SAVE_STEPS=10000
export SEED=1

python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 run_ner.py \
--task_type NER \
--data_dir /home/v-yifeili/blob/dkibdm/aml_components/deberta_verifier_training_stepwise/data \
--labels /home/v-yifeili/blob/dkibdm/aml_components/deberta_verifier_training_stepwise/data/labels.txt \
--model_name_or_path /home/v-yifeili/blob/dkibdm/models_cache/deberta-v3-large \
--output_dir /home/v-yifeili/blob/dkibdm/math_word_problems/debug/deberta_token_level_stepwise/5epochs_lr1e-5_seed1_bs8 \
--max_seq_length 512 \
--num_train_epochs 5 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 256 \
--save_strategy epoch \
--evaluation_strategy no \
--learning_rate 1e-5 \
--lr_scheduler_type constant \
--seed 1 \
--do_eval \
--logging_steps 10 \
--overwrite_output_dir \
--solution_correct_loss_weight 1.0 \
--solution_incorrect_loss_weight 1.0 \
--step_correct_loss_weight 0.1 \
--step_incorrect_loss_weight 0.1 \
--other_label_loss_weight 0.1 \
--num_of_multi_task_solution_wise_and_stepwise_both_train_epochs 2 \
--num_of_solution_wise_only_train_epochs 2