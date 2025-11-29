
python inference.py \
    --image_dir [IMG_PATH] \
    --out_dir ./output_forward \
    --traj_txt test/trajs/loop3.txt \
    --mode 'single_view_1drc_iterative' \
    --center_scale 1. \
    --elevation=5 \
    --seed 123 \
    --d_theta 0 0 \
    --d_phi 0 0 \
    --d_r 0.2 0.2  \
    --d_x 0 0  \
    --d_y 0 0  \
    --ckpt_path ./checkpoints/model.ckpt \
    --config configs/inference_pvd_1024.yaml \
    --ddim_steps 50 \
    --video_length 25 \
    --device 'cuda:3' \
    --height 576 --width 1024 \
    --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth


python inference.py \
    --image_dir [IMG_PATH] \
    --out_dir ./output_y \
    --traj_txt test/trajs/loop3.txt \
    --mode 'single_view_target' \
    --center_scale 1. \
    --elevation=5 \
    --seed 123 \
    --d_theta 0 \
    --d_phi 0 \
    --d_r 0 \
    --d_x 0 \
    --d_y 50  \
    --ckpt_path ./checkpoints/model.ckpt \
    --config configs/inference_pvd_1024.yaml \
    --ddim_steps 50 \
    --video_length 25 \
    --device 'cuda:4' \
    --height 576 --width 1024 \
    --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth


python inference.py \
    --image_dir [IMG_PATH] \
    --out_dir ./output_x \
    --traj_txt test/trajs/loop3.txt \
    --mode 'single_view_ref_iterative' \
    --center_scale 1. \
    --elevation=5 \
    --seed 123 \
    --d_theta 0 \
    --d_phi 0 \
    --d_r 0 \
    --d_x 25 \
    --d_y 0  \
    --ckpt_path ./checkpoints/model.ckpt \
    --config configs/inference_pvd_1024.yaml \
    --ddim_steps 50 \
    --video_length 25 \
    --device 'cuda:5' \
    --height 576 --width 1024 \
    --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth


python inference.py \
    --image_dir [IMG_PATH] \
    --out_dir ./output_rot \
    --traj_txt test/trajs/loop3.txt \
    --mode 'single_view_ref_iterative' \
    --center_scale 1. \
    --elevation=5 \
    --seed 456 \
    --d_theta 0 0 \
    --d_phi -30 30 \
    --d_r 0.1 0.1 \
    --d_x 0  \
    --d_y 0  \
    --ckpt_path ./checkpoints/model.ckpt \
    --config configs/inference_pvd_1024.yaml \
    --ddim_steps 50 \
    --video_length 25 \
    --device 'cuda:6' \
    --height 576 --width 1024 \
    --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

python inference.py \
    --image_dir [IMG_PATH] \
    --out_dir ./output_pitch_a \
    --traj_txt test/trajs/loop3.txt \
    --mode 'single_view_target' \
    --center_scale 1. \
    --elevation=5 \
    --seed 123 \
    --d_theta 15 \
    --d_phi 0 \
    --d_r 0.1 \
    --d_x 0 \
    --d_y 0 \
    --ckpt_path ./checkpoints/model.ckpt \
    --config configs/inference_pvd_1024.yaml \
    --ddim_steps 50 \
    --video_length 25 \
    --device 'cuda:7' \
    --height 576 --width 1024 \
    --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth