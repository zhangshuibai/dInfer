# Expert-choice support notes (modeling_llada2_moe_sglang.py)

Goal: add expert-choice routing (experts pick tokens) to
`python/dinfer/model/modeling_llada2_moe_sglang.py`, which currently uses
token-choice (tokens pick experts).

Note: please create a new branch for this work; do not modify `master`.

## Current token-choice path
- Routing happens in `LLaDA2SparseMoeBlock._forward_router_experts()`:
  - `router_logits = gate(hidden_states)` gives `[N, E]`.
  - `TopK` selects top-K experts per token -> `topk_idx/topk_weights` as `[N, K]`.
  - `FusedMoE(hidden_states, topk_output)` executes experts and merges outputs.
- DeepEP path (`forward_deepep`) assumes token-choice and uses
  `DeepEPDispatcher.dispatch/combine` with token->expert mapping.

## What must change for expert-choice
1) **Routing strategy switch**
   - Add a config flag (e.g., `routing_strategy = "expert_choice"`) and branch
     inside `LLaDA2SparseMoeBlock` to select expert-choice vs token-choice.
   - Reason: expert-choice requires different selection and data layout.

2) **Router selection**
   - Replace or extend `TopK` so it can pick tokens per expert.
     - Input still `router_logits: [N, E]`.
   - Output must be expert-centric (e.g., per-expert token lists).
   - Reason: token-choice output `[N, K]` does not encode capacity or
     per-expert token lists required by expert-choice.

3) **Expert execution interface**
   - `FusedMoE` currently expects token-choice inputs.
   - Either:
     - Extend `FusedMoE` to accept expert-centric routing (token lists per
       expert + weights), or
     - Insert a reshaping/dispatch layer that converts expert-choice outputs
       into the format expected by `FusedMoE`.
   - Reason: expert-choice changes how tokens are grouped and fed to experts.

4) **DeepEP (A2A) path**
   - Update `forward_deepep()` and `DeepEPDispatcher` usage.
   - With expert-choice, the dispatch should be expert->token centered, not
     token->expert. The dispatcher API and its inputs will need to match the
     new routing outputs.
   - Reason: current dispatch assumes token-choice and will misroute tokens.

## Minimal code touch points
- `python/dinfer/model/modeling_llada2_moe_sglang.py`
  - `LLaDA2SparseMoeBlock.__init__`: add routing strategy flag, select
    router implementation.
  - `LLaDA2SparseMoeBlock._forward_router_experts`: replace token-choice
    routing with expert-choice branch.
  - `LLaDA2SparseMoeBlock.forward_deepep`: update dispatch/combine inputs.
- In config:
  - `python/dinfer/model/configuration_llada2_moe.py` (or related config)
    to expose the routing strategy flag.

## Notes
- If SGLang already provides an expert-choice router or a compatible
  `FusedMoE` interface, prefer integrating that instead of re-implementing
  from scratch.
- You will likely need new tests for routing correctness and capacity
  behavior, especially in the DeepEP path.

## Download + debug
- Please see `/data/dInfer/README.md` to configure the environment.
- Download the model to a local directory (example path used below):
  `huggingface-cli download inclusionAI/LLaDA2.0-mini-preview --local-dir /data/models/LLaDA2.0-mini-preview`
- For debugging/evaluation, run the script:
  `/data/dInfer/evaluations/eval_llada_mini.sh`
