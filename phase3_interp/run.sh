
# example 1
export sel="flying_car"
export render_folder="../static/render_images/$sel/"
export keyframe_folder="../phase2_keyframes/outputs/frames/$sel/"
export save_folder="outputs/video/$sel"
export inverted_folder="outputs/latents/$sel"

# # example 2
# export sel="swimming_duck"
# export render_folder="../static/render_images/$sel/"
# export keyframe_folder="../phase2_keyframes/outputs/frames/$sel/"
# export save_folder="outputs/video/$sel"
# export inverted_folder="outputs/latents/$sel"

# # example 3
# export sel="christmas_dog"
# export render_folder="../static/render_images/$sel/"
# export keyframe_folder="../phase2_keyframes/outputs/frames/$sel/"
# export save_folder="outputs/video/$sel"
# export inverted_folder="outputs/latents/$sel"

# # example 4
# export sel="space_stormtrooper"
# export render_folder="../static/render_images/$sel/"
# export keyframe_folder="../phase2_keyframes/outputs/frames/$sel/"
# export save_folder="outputs/video/$sel"
# export inverted_folder="outputs/latents/$sel"

python gen_video.py --gpu 0 \
    --render_folder $render_folder \
    --keyframe_folder $keyframe_folder \
    --inverted_folder $inverted_folder \
    --save_folder $save_folder \
    --use_vis \
    --use_controlnet
