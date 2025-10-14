import argparse
import os
import torch
from PIL import Image
import numpy as np
from utils.models.inversion.edm_svd import SVDInbetween
from utils.models.injection.injector import init_pnp
from utils.models.injection.pnp_utils import register_spatial_attn_masks
from utils.utils.load_utils import normalize_depth, frames_to_video, prepare_masks, save_concat

from svd.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from svd.models.controlnet.controlnet_svd import ControlNetSVDModel
from svd.pipelines.pipeline_stable_video_diffusion_controlnet import StableVideoDiffusionPipelineControlNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--pnp_spatial_attn_t", type=float, default=1.0)
    parser.add_argument("--pnp_f_t", type=float, default=1.0)
    parser.add_argument("--render_folder", type=str)
    parser.add_argument("--keyframe_folder", type=str)
    parser.add_argument("--inverted_folder", type=str)
    parser.add_argument("--save_folder", type=str)
    parser.add_argument('--skip_inversion', action="store_true")
    parser.add_argument('--use_controlnet', action="store_true")
    parser.add_argument('--use_vis', action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    return args

def load_svd():
    device = torch.device("cuda")
    data_type = torch.float16
    
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        subfolder="unet",
        low_cpu_mem_usage=False,
        torch_dtype=data_type,
    )
    
    pretrained_model_name_or_path = "stabilityai/stable-video-diffusion-img2vid"        
    controlnet = ControlNetSVDModel.from_pretrained("CiaraRowles/temporal-controlnet-depth-svd-v1", subfolder="controlnet").to(device, dtype=torch.float16)
    model = StableVideoDiffusionPipelineControlNet.from_pretrained(pretrained_model_name_or_path,controlnet=controlnet,unet=unet).to(device, dtype=torch.float16)
    print("start loading Diffusion Model...")
    return model

def invert(invertor, model, source_frames, invert_folder, fps, depth_frames=None, debug_mode=False):
    first_frame = source_frames[0]
    width, height = first_frame.size
    latents_inv = invertor.get_inverted_latent(
                    pipe=model,
                    src_images=source_frames,
                    depth_frames=depth_frames,
                    width=width, 
                    height=height,
                    num_frames=len(depth_frames),
                    target_fps=fps
                ) 
    
    latents_inv = latents_inv[::-1]
    os.makedirs(invert_folder, exist_ok=True)
    torch.save(latents_inv, f"{invert_folder}/latents_inv.pt")

    if debug_mode:
        latents = latents_inv[0]
        frames = invertor(
                pipe=model,
                first_frame=first_frame,
                depth_frames=depth_frames,
                latents=latents,
                width=width, 
                height=height,
                num_frames=len(depth_frames)
            ) 
        
        frames_save_folder = os.path.join(invert_folder, "frames")
        os.makedirs(frames_save_folder, exist_ok=True)
        frames_to_video(frames, f"{frames_save_folder}/video.mp4")
    
def gen_video(pipe, width, height, fps, path_dict):
    # load condition images
    first_frame = load_image(path_dict["first_frame_path"]) 
    end_frame = load_image(path_dict["end_frame_path"])
    rendered_start = load_image(path_dict["first_render_path"])
    rendered_end = load_image(path_dict["end_render_path"])

    # get the inverted forward and reverse latents
    latents_inv = torch.load(path_dict["first_latent_path"])
    reverse_latents_inv = torch.load(path_dict["end_latent_path"])
    start_code = latents_inv[0]
    
    # get depth frames
    if path_dict["depth_paths"] is None:
        depth_frames = None
    else:
        depth_frames = [normalize_depth(path, w=width, h=height) for path in path_dict["depth_paths"]]

    # prepare the visibility mask
    vis_paths = path_dict["vis_paths"]
    if vis_paths != None:
        visibility_frames = [Image.open(path).resize((width, height)) for path in vis_paths]
        vis_masks = prepare_masks(visibility_frames)
        
        rm_k = max(list(vis_masks.keys()))
        erase_mask = vis_masks[rm_k].unsqueeze(1).unsqueeze(0)
        randn_noise = torch.randn_like(start_code)
        start_code = torch.where(erase_mask, start_code, randn_noise)
        
        vis_masks = {k: vis_masks[k].view(-1, k).unsqueeze(-1) for k in vis_masks}
        register_spatial_attn_masks(pipe, vis_masks)

    frames = InterpPnP.sample_with_pnp(
                pipe=pipe,
                first_frame=first_frame, 
                end_frame=end_frame, 
                src_images=[rendered_start, rendered_end],
                depth_frames=depth_frames,
                start_code = start_code,
                first_latents_inv = latents_inv,
                end_latents_inv = reverse_latents_inv,
                width=width,
                height=height,
                target_fps=fps
            ) 

    # save batch frames
    save_path = path_dict["save_path"]
    os.makedirs(save_path, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(f"{save_path}/{i}.png")

    return frames

if __name__ == "__main__":
    args = parse_args()
    load_image = lambda path, size=(args.width, args.height): Image.open(path).convert("RGB").resize(size)
    
    # Step 1: get the number of keyframes
    num_key_frames = len(os.listdir(args.keyframe_folder)) - 1 # -1 for debug.jpg
    keyframes_idx_arr = [1 + (args.window_size - 1) * i for i in range(num_key_frames)]

    # Step 2: initialize model
    model = load_svd()
    InterpPnP = SVDInbetween()
    
    # Step 3: invert the frames
    for keyframe_i in keyframes_idx_arr[:-1]:
        inverted_folder = f"{args.inverted_folder}/{keyframe_i}"
        if not os.path.exists(inverted_folder):
            args.skip_inversion = False
        if not args.skip_inversion:
            # forward
            print(f"Invert frames {keyframe_i} to {keyframe_i+args.window_size-1} ...")
            source_frames = [load_image(f"{args.render_folder}/ani/{keyframe_i+i:04d}.png") \
                                                        for i in range(args.window_size)]
            if args.use_controlnet:
                depth_paths = [f"{args.render_folder}/depth/{keyframe_i+i:04d}.png" for i in range(args.window_size)]
                depth_frames = [normalize_depth(path, w=args.width, h=args.height) for path in depth_paths]
            else:
                depth_frames = None
            
            invert(InterpPnP, model, source_frames, inverted_folder, args.fps, depth_frames=depth_frames, debug_mode=False)
            
            # reverse
            print(f"Invert frames {keyframe_i+args.window_size-1} to {keyframe_i} ...")
            inverted_folder = f"{args.inverted_folder}/{keyframe_i}_reverse"
            invert(InterpPnP, model, source_frames[::-1], inverted_folder, args.fps, depth_frames=depth_frames, debug_mode=False)

    
    # Step 4: Video Generation
    # 4.1 Turn on Feature Injection
    init_pnp(model, pnp_f_t=args.pnp_f_t, pnp_spatial_attn_t=args.pnp_spatial_attn_t)

    all_frames = []
    for keyframe_i in keyframes_idx_arr[:-1]:
        keyframe_j = keyframe_i+args.window_size-1
        i = (keyframe_i - 1) // (args.window_size - 1)
        path_dict = {
            "first_frame_path": f"{args.keyframe_folder}/{i:03d}.png",
            "end_frame_path": f"{args.keyframe_folder}/{i+1:03d}.png",
            "first_latent_path": f"{args.inverted_folder}/{keyframe_i}/latents_inv.pt",
            "end_latent_path": f"{args.inverted_folder}/{keyframe_i}_reverse/latents_inv.pt",
            "first_render_path": f"{args.render_folder}/ani/{keyframe_i:04d}.png",
            "end_render_path": f"{args.render_folder}/ani/{keyframe_j:04d}.png",
            "vis_paths": [f"{args.render_folder}/mask/{j:04d}.png" for j in range(keyframe_i, keyframe_j+1)] if args.use_vis else None,
            "depth_paths": [f"{args.render_folder}/depth/{j:04d}.png" for j in range(keyframe_i, keyframe_j+1)] if args.use_controlnet else None,
            "save_path": f"{args.save_folder}/{keyframe_i}",
        }

        frames = gen_video(model, width=args.width, height=args.height, fps=args.fps, path_dict=path_dict)
        
        if keyframe_i == keyframes_idx_arr[-2]: # this is last batch
            all_frames += frames
        else:
            all_frames += frames[:-1]
        
    save_video_name = os.path.join(args.save_folder, "gen_video.mp4")
    frames_to_video(all_frames, save_video_name, fps=24)
    

    source_frames = [load_image(f"{args.render_folder}/ani/{i:04d}.png") \
                                                        for i in range(keyframes_idx_arr[0], keyframes_idx_arr[-1]+1)]
    render_video_name = os.path.join(args.save_folder, "render.mp4")
    frames_to_video(source_frames, render_video_name, fps=24)

    concat_video_name = os.path.join(args.save_folder, "concat.mp4")    
    save_concat(source_frames, all_frames, concat_video_name, fps=24)
