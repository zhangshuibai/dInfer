# ParallelBench Integration for dInfer

This directory contains the integration of ParallelBench tasks into dInfer's evaluation framework.

## Structure

- `adapter.py`: Adapter functions to convert ParallelBench format to lm-eval-harness format
- `metrics.py`: Copied from ParallelBench for metric calculations
- `waiting_line/`, `puzzle/`, etc.: YAML configuration files for each task

## Usage

Use these tasks with dInfer's eval scripts:

```bash
cd /data/dInfer/evaluations
python eval_dinfer.py \
  --tasks waiting_line_copy \
  --confirm_run_unsafe_code \
  --model dInfer_eval \
  --model_args model_path=YOUR_MODEL_PATH,gen_length=32,block_length=1,... \
  --output_path ./outputs \
  --include_path ./tasks/parallel_bench \
  --apply_chat_template
```

## Available Tasks

### Waiting Line Tasks
- `waiting_line_copy`
- `waiting_line_reverse`
- `waiting_line_sort`
- `waiting_line_shuffle`

### Puzzle Tasks
- `puzzle_sudoku_n4_12`

### Paraphrase/Summarize Tasks
- `paraphrase_summarize_samsum`

## Notes

- Task names are automatically inferred from file paths
- Metrics are automatically selected based on task type
- Few-shot examples from ParallelBench are preserved in the prompts

