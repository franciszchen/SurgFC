export LLAVA_DEFAULT_CONVERSATION="conv_vicuna_v1"

# run
export LOGDIR=logs/
export out_dir="${LOGDIR}/llava-plus/llava_plus_v1.3_7b_0717"
mkdir -p $out_dir
echo ${out_dir}/loginfo.txt

# Note: Our scripts support multi-source data and image folders. Seperate each item with `,`. Note that it may cause problems if multiple folders have images with the same name.

deepspeed --include localhost:4,5,6,7 --master_port 25641 /mnt/xingjian_luo/project/LLaVA-Plus/llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path  /mnt/xingjian_luo/checkpoint/llava_plus \
    --mm_projector_type mlp2x_gelu \
    --version v1 \
    --data_path /mnt/xingjian_luo/dataset/surgFC/dataset/ano_det/ano_det.json,/mnt/xingjian_luo/dataset/surgFC/dataset/ano_seg/ano_seg.json,/mnt/xingjian_luo/dataset/surgFC/dataset/depth/depth_model.json,/mnt/xingjian_luo/dataset/surgFC/dataset/reconstruct/reconstruct_model.json,/mnt/xingjian_luo/dataset/surgFC/dataset/skill/skill_model.json,/mnt/xingjian_luo/dataset/surgFC/dataset/tool_det/detect_500_wales.json,/mnt/xingjian_luo/dataset/surgFC/dataset/tool_seg/segment_1000_wales.json,/mnt/xingjian_luo/dataset/surgFC/dataset/triplet/triplet_tool_614.json \
    --image_folder /mnt/xingjian_luo/dataset/llava_plus_dataset/coco,/mnt/xingjian_luo/dataset/llava_plus_dataset/xiehe,/mnt/xingjian_luo/dataset/llava_plus_dataset/wales,/mnt/xingjian_luo/dataset/llava_plus_dataset/wales_extra \
    --vision_tower /mnt/xingjian_luo/checkpoint/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $out_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
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