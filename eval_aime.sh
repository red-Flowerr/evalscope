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
    --model /mnt/hdfs/tiktok_aiic/user/codeai/hf_models/gpt-oss-20b \
    --tensor-parallel-size 4 \
    --port 8801