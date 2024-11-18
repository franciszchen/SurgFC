export LLAVA_DEFAULT_CONVERSATION="conv_vicuna_v1"

# run
export LOGDIR=logs/
export out_dir="${LOGDIR}/llava-plus/mmprojector_0405_4experts_no_noise_debug"
mkdir -p $out_dir
echo ${out_dir}/loginfo.txt

# Note: Our scripts support multi-source data and image folders. Seperate each item with `,`. Note that it may cause problems if multiple folders have images with the same name.

deepspeed --include localhost:4,5,6,7 --master_port 25649 /home/lxj/project/LLaVA-Plus/llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path /home/lxj/project/FastChat/vicuna_7b_med_0118/checkpoint-2400 \
    --version plain \
    --data_path /home/lxj/dataset/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json,/home/lxj/dataset/LLaVA-Pretrain/pretrain_surgical_94k_0305.json \
    --image_folder /home/lxj/dataset/LLaVA-Pretrain/images \
    --vision_tower /home/lxj/checkpoint/llava_checkpoint/clip-vit-large-patch14 \
    --mm_projector_type moe \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $out_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
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







