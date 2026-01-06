from evalscope import TaskConfig, run_task

from dotenv import dotenv_values
env = dotenv_values('.env')
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='/mnt/hdfs/tiktok_aiic/user/codeai/hf_models/Qwen3-14B',
    api_url='http://127.0.0.1:8801/v1/chat/completions',
    eval_type='openai_api',
    datasets=['live_code_bench'],# ['humaneval'], # ['live_code_bench'],
    eval_batch_size=128,
    generation_config={
        'max_tokens': 8192,
        'temperature': 0.0,
        'seed': 42,
    },
    # use_sandbox=True, # 启用沙箱
    # sandbox_type='docker', # 指定沙箱类型
    # sandbox_manager_config={
    #     'base_url': 'http://10.251.231.210:1234',  # 远端沙箱管理器URL
    #     'headers': {'Destination-Service': 'ms-enclave-sandbox'}
    # },
    # judge_worker_num=5, # 指定评测时的沙箱工作进程数
    stream=True,  # 是否使用流式输出
    limit=2,  # 设置为100条数据进行测试
    observe_entropy=True,
    entropy_top_logprobs=5
)

run_task(task_cfg=task_cfg)