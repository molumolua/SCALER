
#!/usr/bin/env bash
set -euo pipefail

SANDBOX_URL="http://localhost:8080/run_code"
# Loading Args
MAX_ROWS=-1
LOAD_DIR="/inspire/hdd/global_user/xucaijun-253108120121/Dataset/hf_datasets/code_contest"
LOAD_TYPE="parquet"
FILE_GLOB="train-*.parquet"
GENERATOR_FILE_GLOB="with_generator_after_filter.jsonl"
# Save Args
SAVE_DIR="/inspire/hdd/global_user/xucaijun-253108120121/Code/FORGE_THIRD"
FILTER_PROBLEM_SAVE_NAME="after_filter.jsonl"
FILTER_PROBLEM_META_NAME="after_filter.json"
GENERATOR_SAVE_NAME="with_generator_after_filter.jsonl"
GENERATOR_META_NAME="with_generator_after_filter.json"


TRAIN_SAVE_NAME="FORGE_train_data_large.json"
TRAIN_META_NAME="FORGE_train_data_large_meta.json"



# API Args
# MODEL="gpt-5-mini-2025-08-07"
MODEL="glm-4.6"
TEMPERATURE=0.6
N_PROCESSES=20


# Verify group logic problem
CHECK_NUMBER_FOR_VERIFY_PROBLEMS=3
DEEP_TEST_TIMES=100
BREADTH_TEST_TIMES=100
DIFFERENT_OUTPUT_LIMIT=10
MAX_OUTPUT_RATE=0.3


# Train args
MAX_PROMPT_LENGTH=2048
TRAIN_MODEL_PATH="/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B"
BATCH_SIZE=512

DIFFERENT_OUTPUT_LIMIT=10
MAX_OUTPUT_RATE=0.3

# python api_filter_problem_for_environment.py \
#  --load_type ${LOAD_TYPE} \
#  --load_dir ${LOAD_DIR} \
#  --file_glob ${FILE_GLOB} \
#  --max_rows ${MAX_ROWS} \
#  --save_dir ${SAVE_DIR} \
#  --save_name ${FILTER_PROBLEM_SAVE_NAME} \
#  --save_meta_name ${FILTER_PROBLEM_META_NAME} \
#  --model ${MODEL}\
#  --batch_size ${BATCH_SIZE}


 python api_generate_generator_for_environment.py \
  --load_type json \
  --load_dir ${SAVE_DIR} \
  --file_glob ${FILTER_PROBLEM_SAVE_NAME} \
  --max_rows ${MAX_ROWS} \
  --save_dir ${SAVE_DIR} \
  --save_name ${GENERATOR_SAVE_NAME} \
  --save_meta_name ${GENERATOR_META_NAME} \
  --model ${MODEL} \
  --check_number ${CHECK_NUMBER_FOR_VERIFY_PROBLEMS} \
  --sandbox_url ${SANDBOX_URL} \
  --batch_size ${BATCH_SIZE} \
  --n_processes ${N_PROCESSES} \
  --different_output_limit ${DIFFERENT_OUTPUT_LIMIT} \
  --max_output_rate ${MAX_OUTPUT_RATE} \
  --deep_test_times ${DEEP_TEST_TIMES} \
  --breadth_test_times ${BREADTH_TEST_TIMES} \



#  python set_max_difficulty_and_process_train_data.py \
#  --load_type json \
#  --load_dir ${SAVE_DIR} \
#  --file_glob ${GENERATOR_FILE_GLOB} \
#  --max_rows ${MAX_ROWS} \
#  --save_dir ${SAVE_DIR} \
#  --save_name ${TRAIN_SAVE_NAME} \
#  --save_meta_name ${TRAIN_META_NAME} \
#  --model ${MODEL}\
#  --max_prompt_length ${MAX_PROMPT_LENGTH} \
#  --sandbox_url ${SANDBOX_URL} \
#  --batch_size ${BATCH_SIZE} \
#  --n_processes ${N_PROCESSES} 

