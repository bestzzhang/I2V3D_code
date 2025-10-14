
export GPU=0
export window_size=16
export max_batch_size=4
export overlap_size=1

# example 1
export SEL="flying_car"
export save_latents_dir="outputs/latents/$SEL"
export save_frames_dir="outputs/frames/$SEL"
export data_path="../static/render_images/$SEL"
export lora_path="../static/lora/$SEL/pytorch_lora_weights.safetensors"
export prompt="A sks car flying in the sky."

# # example 2
# export SEL="swimming_duck"
# export save_latents_dir="outputs/latents/$SEL"
# export save_frames_dir="outputs/frames/$SEL"
# export data_path="../static/render_images/$SEL"
# export lora_path="../static/lora/$SEL/pytorch_lora_weights.safetensors"
# export prompt="A sks duck wearing a diving suit plays on a slide."

# # example 3
# export SEL="christmas_dog"
# export save_latents_dir="outputs/latents/$SEL"
# export save_frames_dir="outputs/frames/$SEL"
# export data_path="../static/render_images/$SEL"
# export lora_path="../static/lora/$SEL/pytorch_lora_weights.safetensors"
# export prompt="A sks dog wearing reindeer antlers and a bell around its neck. The dog is in front of a sled filled with colorful gift boxes."

# # example 4
# export SEL="space_stormtrooper"
# export save_latents_dir="outputs/latents/$SEL"
# export save_frames_dir="outputs/frames/$SEL"
# export data_path="../static/render_images/$SEL"
# export lora_path="../static/lora/$SEL/pytorch_lora_weights.safetensors"
# export prompt="A sks stormtrooper in the space."


python gen_keyframes.py --data_path $data_path \
                    --save_latents_dir $save_latents_dir \
                    --save_frames_dir $save_frames_dir \
                    --lora_path $lora_path \
                    --prompt "$prompt" \
                    --gpu $GPU \
                    --max_batch_size $max_batch_size \
                    --video_length $window_size \
                    --overlap_size $overlap_size \
                    --conv_t 0.6

