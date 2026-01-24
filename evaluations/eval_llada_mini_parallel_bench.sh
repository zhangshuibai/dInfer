# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
source /data/miniconda3/etc/profile.d/conda.sh
conda activate dinfer

parallel_decoding='threshold' # or hierarchy
length=64 # generate length (ParallelBench tasks typically need 32-64 tokens)
block_length=32 # block length
model_path='/data/dFactory/llada2_mini_bd_parallelbench_mixture_sft_outputs/hf_ckpt_split' # your model path
threshold=0.80 # threshold for parallel decoding
low_threshold=0.62 # low threshold for parallel decoding when using hierarchy mechanism
cache='prefix' # or 'prefix' for prefix cache; or '' if you don't want to use cache
warmup_times=0 # warmup times for cache
prefix_look=0
after_look=0
cont_weight=0 # cont weight
use_credit=False # enable credit for threshold mechanism
use_compile=True # use compile
tp_size=4 # tensor parallel size (using 4 GPUs)
gpus='0' # gpus for tensor parallel inference (mapped to physical GPUs 4,5,6,7 via CUDA_VISIBLE_DEVICES)
parallel='tp' # 'tp' for tensor parallel or 'dp' for data parallel
output_dir='./outputs' # your customer output path
model_type='llada2' # llada2 (for llada2-mini) 
use_bd=True # use block diffusion
master_port="23457"
save_samples=True # save samples (set to True for debugging)
#waiting_line_copy waiting_line_reverse waiting_line_shuffle waiting_line_sort
#waiting_line_insert_index waiting_line_insert_random waiting_line_remove_index waiting_line_remove_random waiting_line_replace_index waiting_line_replace_random

# Extract model name from model_path
# Example: /data/dFactory/llada2_mini_bd_parallelbench_mixture_sft_outputs/hf_ckpt_split
# -> llada2_mini_bd_parallelbench_mixture_sft_outputs -> llada2-mixture
model_name=$(basename $(dirname ${model_path}))
# Simplify model name: remove common suffixes like _sft_outputs, _hf_ckpt, etc.
model_name=$(echo ${model_name} | sed 's/_sft_outputs$//' | sed 's/_hf_ckpt$//' | sed 's/_outputs$//')
# Replace underscores with hyphens for cleaner folder names
model_name=$(echo ${model_name} | sed 's/_/-/g')
# Simplify long names: extract key parts (e.g., llada2-mini-bd-parallelbench-mixture -> llada2-mixture)
if [[ ${model_name} == *"llada2"*"mixture"* ]]; then
  model_name="llada2-mixture"
elif [[ ${model_name} == *"llada2"*"mini"* ]]; then
  model_name="llada2-mini"
fi

# ParallelBench tasks: waiting_line tasks, puzzle tasks, paraphrase_summarize tasks
if [ "${parallel}" = "tp" ]; then
  # All waiting_line tasks in test-aug directory
  for task in waiting_line_copy waiting_line_reverse waiting_line_shuffle waiting_line_sort \
             waiting_line_insert_index waiting_line_insert_random \
             waiting_line_remove_index waiting_line_remove_random \
             waiting_line_replace_index waiting_line_replace_random; do
    # Format threshold: replace dot with underscore for folder name (e.g., 0.80 -> 0_80)
    threshold_str=$(echo ${threshold} | tr '.' '_')
    # Loop through enable_remask values (True and False)
    for enable_remask in True False; do
      output_path=${output_dir}/${model_name}/${task}_${parallel_decoding}_${threshold_str}_remask_${enable_remask}
      /data/miniconda3/envs/dinfer/bin/python eval_dinfer_sglang.py --tasks ${task} \
      --confirm_run_unsafe_code --model dInfer_eval \
      --model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},show_speed=True,save_dir=${output_path},parallel_decoding=${parallel_decoding},cache=${cache},warmup_times=${warmup_times},use_compile=${use_compile},tp_size=${tp_size},parallel=${parallel},cont_weight=${cont_weight},use_credit=${use_credit},prefix_look=${prefix_look},after_look=${after_look},gpus=${gpus},model_type=${model_type},use_bd=${use_bd},master_port=${master_port},save_samples=${save_samples},enable_remask=${enable_remask} \
      --output_path ${output_path} --include_path "$(pwd)/tasks/parallel_bench" --apply_chat_template
    done
  done
else
  echo "parallel must be tp"
fi
