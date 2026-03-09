# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES=3
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/../python:${PYTHONPATH}"

parallel_decoding='threshold' # or hierarchy
length=256 # generate length (40GB single-card ultra-safe default)
block_length=64 # block length (further reduce KV/cache peak)
model_path='/workspace/dInfer/models/LLaDA2.0-mini-preview' # your model path
threshold=0.80 # threshold for parallel decoding
low_threshold=0.62 # low threshold for parallel decoding when using hierarchy mechanism
cache='prefix' # keep KV cache enabled
warmup_times=0 # warmup times for cache
prefix_look=0
after_look=0
cont_weight=0 # cont weight
use_credit=False # enable credit for threshold mechanism
use_compile=False # disable compile to reduce extra runtime memory overhead
tp_size=1 # tensor parallel size for single GPU
gpus='0' # logical GPU id after CUDA_VISIBLE_DEVICES remap
parallel='tp' # 'tp' for tensor parallel or 'dp' for data parallel
output_dir='/volume/demo/xlzhuang/moeinference/dInfer/evaluations/outputs' # your customer output path
model_type='llada2' # llada2 (for llada2-mini) 
use_bd=False # disable BlockDiffusion to reduce peak memory on 40GB single-card
master_port="23456"
save_samples=False # save samples
routing_strategy='token_choice' # 'token_choice' (default) or 'expert_choice' for Expert Choice routing
expert_capacity='' # capacity per expert for expert_choice (leave empty for auto: n*top_k/num_experts)
profile_experts=True # whether to output layer-internal expert straggler profile
profile_mode='exact_torch' # estimate | exact_torch
profile_output_dir="${output_dir}/profile_${routing_strategy}" # csv output dir for profiling
limit=5 # number of samples to run (single-card profiling default)
# for llada 1.5 use tasks gsm8k_llada1.5 mbpp_sanitized_llada1.5
# for llada2_mini use tasks gsm8k_llada_mini mbpp_sanitized_llada_mini
if [ "${parallel}" = "tp" ]; then
  for task in gsm8k_llada_mini; do
    output_path=${output_dir}/${task}
    python eval_dinfer_sglang.py --tasks ${task} \
    --confirm_run_unsafe_code --model dInfer_eval \
    --model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},show_speed=True,save_dir=${output_path},parallel_decoding=${parallel_decoding},cache=${cache},warmup_times=${warmup_times},use_compile=${use_compile},tp_size=${tp_size},parallel=${parallel},cont_weight=${cont_weight},use_credit=${use_credit},prefix_look=${prefix_look},after_look=${after_look},gpus=${gpus},model_type=${model_type},use_bd=${use_bd},master_port=${master_port},save_samples=${save_samples},routing_strategy=${routing_strategy},profile_experts=${profile_experts},profile_mode=${profile_mode},profile_output_dir=${profile_output_dir}${expert_capacity:+,expert_capacity=${expert_capacity}} \
    --output_path ${output_path} --include_path "$(pwd)/tasks" --apply_chat_template ${limit:+--limit ${limit}}
  done
else
  echo "parallel must be tp"
fi
