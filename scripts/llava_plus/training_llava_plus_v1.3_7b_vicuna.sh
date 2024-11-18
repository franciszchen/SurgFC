export LLAVA_DEFAULT_CONVERSATION="conv_vicuna_v1"

# run
export LOGDIR=logs/
export out_dir="${LOGDIR}/llava-plus/llava_plus_v1.3_7b_0119"
mkdir -p $out_dir
echo ${out_dir}/loginfo.txt

# Note: Our scripts support multi-source data and image folders. Seperate each item with `,`. Note that it may cause problems if multiple folders have images with the same name.

deepspeed --include localhost:4,5,6,7 /home/lxj/project/LLaVA-Plus/llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path /home/lxj/project/FastChat/vicuna_7b_med_0118/checkpoint-2400 \
    --pretrain_mm_mlp_adapter /home/lxj/project/LLaVA-Plus/logs/llava-plus/mmprojector_0119/mm_projector.bin \
    --version v1 \
    --mm_projector_type mlp2x_gelu \
    --data_path /home/lxj/dataset/llava_plus_dataset/vicuna_json/no_tool_40k.json,/home/lxj/dataset/llava_plus_dataset/vicuna_json/surgical_total_5k.json,/home/lxj/dataset/llava_plus_dataset/vicuna_json/tool_9k_seg.json,/home/lxj/dataset/llava_plus_dataset/vicuna_json/tool_19k_det.json\
    --image_folder /home/lxj/dataset/llava_plus_dataset/coco,/home/lxj/dataset/llava_plus_dataset/xiehe,/home/lxj/dataset/llava_plus_dataset/wales  \
    --vision_tower /home/lxj/checkpoint/llava_checkpoint/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $out_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    >> ${out_dir}/loginfo.txt 2>&1