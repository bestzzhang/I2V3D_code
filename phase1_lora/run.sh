export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export GPU=0
export RANK=32

# example 1
export PROMPT="A sks car flying in the sky."
export SEL="flying_car"
export CLASS_DATA_DIR="../static/multi_views/$SEL"
export INSTANCE_DIR="../static/train_images/$SEL"
export OUTPUT_DIR="lora/$SEL/"

# # example 2
# export PROMPT="A sks duck wearing a diving suit plays on a slide."
# export SEL="swimming_duck"
# export CLASS_DATA_DIR="../static/multi_views/$SEL"
# export INSTANCE_DIR="../static/train_images/$SEL"
# export OUTPUT_DIR="lora/$SEL/"

# # example 3
# export PROMPT="A sks dog wearing reindeer antlers and a bell around its neck. The dog is in front of a sled filled with colorful gift boxes."
# export SEL="christmas_dog"
# export CLASS_DATA_DIR="../static/multi_views/$SEL"
# export INSTANCE_DIR="../static/train_images/$SEL"
# export OUTPUT_DIR="lora/$SEL/"

# # example 4
# export PROMPT="A sks stormtrooper in the space."
# export SEL="space_stormtrooper"
# export CLASS_DATA_DIR="../static/multi_views/$SEL"
# export INSTANCE_DIR="../static/train_images/$SEL"
# export OUTPUT_DIR="lora/$SEL/"

CUDA_VISIBLE_DEVICES=$GPU python train_lora_ffn_view_aug.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --pretrained_vae_model_name_or_path=$VAE_PATH \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --instance_prompt="$PROMPT" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --validation_epochs=999 \
    --random_flip \
    --rank $RANK \
    --with_prior_preservation \
    --class_data_dir $CLASS_DATA_DIR