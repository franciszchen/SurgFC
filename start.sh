conda activate llava_plus
cd /home/lxj/project/LLaVA-Plus

python -m llava.serve.controller --host 0.0.0.0 --port 20001

python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:20001 --port 40000 --worker http://localhost:40000 --model-path /home/lxj/project/LLaVA-Plus/logs/llava-plus/VS-Assistant-------------------------------------------------------------------------------------------------------------------llava_plus_v1.3_7b_0131_vicuna_moe_8expert_surgical_noisy
CUDA_VISIBLE_DEVICES=4,5,6,7 python serve/detect_model_worker.py

CUDA_VISIBLE_DEVICES=4,5,6,7 python serve/segment_model_worker.py

python -m llava.serve.gradio_web_server_llava_plus --controller http://localhost:20001 --model-list-mode reload --share