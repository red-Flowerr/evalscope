evalscope eval \
--model /mnt/hdfs/tiktok_aiic/user/codeai/hf_models/Qwen2.5-Math-7B-Instruct \
--eval-type openai_api \
--api-url http://127.0.0.1:8801/v1 \
--api-key EMPTY \
--datasets aime24 \
--generation-config '{"do_sample": true, "max_tokens":2048, "temperature":0.7, "top_p":0.8}' \
--eval-batch-size 4 \
--observe-entropy


python -m vllm.entrypoints.openai.api_server \
    --model /mnt/hdfs/tiktok_aiic/user/codeai/hf_models/Qwen3-14B \
    --tensor-parallel-size 4 \
    --port 8801


python scripts/convert_evalscope_observations.py \
    --input /opt/tiger/swe_ts/evalscope/outputs/20260106_060629/reports/Qwen3-14B/entropy/humaneval/openai_humaneval/observations.jsonl \
    --tokens-output data/humaneval_tokens.jsonl \
    --summaries-output data/humaneval_summaries.jsonl \
    --task-type code \
    --skip-special

python run_rq_analysis.py --tokens data/humaneval_tokens.jsonl --summaries data/humaneval_summaries.jsonl --output-dir analysis/humaneval --focus-token-types identifier operator --fit-logreg
run_scaling_plan.py --summaries data/humaneval_summaries.jsonl --tokens data/humaneval_tokens.jsonl --selective-threshold 1.2
--local-threshold 1.0 --output-dir scaling_plans/humaneval

python scripts/plot_entropy_distribution.py \
--input /opt/tiger/swe_ts/evalscope/outputs/20260106_095427/reports/Qwen3-14B/entropy/humaneval/openai_humaneval/observations.jsonl \
--output-dir /opt/tiger/swe_ts/evalscope/outputs/entropy_plots/humaneval \
--skip-empty

python scripts/plot_entropy_distribution.py \
--input /opt/tiger/swe_ts/evalscope/outputs/20260106_120504/reports/Qwen3-14B/entropy/live_code_bench/release_latest/observations.jsonl \
--output-dir /opt/tiger/swe_ts/evalscope/outputs/entropy_plots/lcb \
--skip-empty