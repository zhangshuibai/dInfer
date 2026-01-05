#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error and exit
set -o pipefail  # Return value of a pipeline is the value of the last command to exit with a non-zero status

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES=6,7

parallel_decoding='threshold' # or hierarchy
length=2048 # generate length
block_length=32 # block length
model_path='/data/models/inclusion_data/separate_expert_model/LLaDA2.0-mini' # your model path  
threshold=0.98 # threshold for parallel decoding
low_threshold=0.62 # low threshold for parallel decoding when using hierarchy mechanism
cache='prefix' # or 'prefix' for prefix cache; or '' if you don't want to use cache
warmup_times=0 # warmup times for cache
prefix_look=0
after_look=0
cont_weight=0 # cont weight
use_credit=False # enable credit for threshold mechanism
use_compile=True # use compile
tp_size=2 # tensor parallel size (should match number of GPUs in gpus)
gpus='0;1' # gpus for tensor parallel inference
parallel='tp' # 'tp' for tensor parallel or 'dp' for data parallel
output_dir='./outputs_exp' # your customer output path
model_type='llada2' # llada2 (for llada2-mini) 
use_bd=True # use block diffusion
# Use a dynamic port to avoid conflicts, or check if port is available first
master_port="23592"  # Change this if port is already in use
save_samples=False # save samples
enable_remask_values=(False) # enable remasking for threshold decoder
# for llada 1.5 use tasks gsm8k_llada1.5 mbpp_sanitized_llada1.5
# for llada2_mini use tasks gsm8k_llada_mini mbpp_sanitized_llada_mini
#tasks: gsm8k_llada_mini minerva_math500 minerva_math_algebra_llada_mini aime24 aime25
if [ parallel=='tp' ]; then
  for enable_remask in "${enable_remask_values[@]}"; do
    for task in minerva_math_algebra_llada_mini; do
      output_path=${output_dir}/${task}_remask_${enable_remask}
      python eval_dinfer_sglang.py --tasks ${task} \
      --confirm_run_unsafe_code --model dInfer_eval \
      --model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},show_speed=True,save_dir=${output_path},parallel_decoding=${parallel_decoding},cache=${cache},warmup_times=${warmup_times},use_compile=${use_compile},tp_size=${tp_size},parallel=${parallel},cont_weight=${cont_weight},use_credit=${use_credit},prefix_look=${prefix_look},after_look=${after_look},gpus=${gpus},model_type=${model_type},use_bd=${use_bd},master_port=${master_port},save_samples=${save_samples},enable_remask=${enable_remask} \
      --output_path ${output_path} --include_path ${SCRIPT_DIR}/tasks --apply_chat_template --limit 100
    done
  done
else
  echo "parallel must be tp"
fi
