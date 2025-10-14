
import os
import torch
import argparse
from utils.utils import seed_everything, concat_images, sliding_window, prepare_depth_cond, prepare_fg_masks, prepare_erase_masks, concatenate_source, process_attn_mask, normalize_depth
from utils.pipelines.pipe_sdxl import ExtendAttnSD
from utils.injection.injector import init_method
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str) 
    parser.add_argument('--save_latents_dir', type=str)
    parser.add_argument('--save_frames_dir', type=str)
    parser.add_argument('--lora_path', type=str)
    parser.add_argument('--prompt', type=str)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vae_batch_size', type=int, default=4)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--overlap_size', type=int, default=1)
    parser.add_argument('--video_length', type=int, default=16)
    parser.add_argument('--skip_inversion', action="store_true")
    parser.add_argument('--conv_t', type=float, default=0.6)
    parser.add_argument('--pnp_attn_fg_t', type=float, default=1.0)
    parser.add_argument('--pnp_attn_bg_t', type=float, default=0.8)
    parser.add_argument('--inversion_prompt', type=str, default='')
    parser.add_argument('--max_size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    seed_everything(args.seed)
    if args.pnp_attn_fg_t > 0 or args.pnp_attn_bg_t > 0:
        args.is_pnp = True
    else:
        args.is_pnp = False
    return args

def prep(args, model, depth_frames, frames, save_name = 'latents.pt', debug_mode=False):
    ret = model.extract_latents(frames=frames,
                                         depth_frames=depth_frames,
                                         batch_size=args.vae_batch_size,
                                         inversion_prompt=args.inversion_prompt,
                                         debug_mode=debug_mode)
    if debug_mode:
        recon_frames, inverted_latents = ret
    else:
        inverted_latents = ret

    os.makedirs(args.save_latents_dir, exist_ok=True)
    latents_save_path = os.path.join(args.save_latents_dir, save_name)
    torch.save(inverted_latents[::-1], latents_save_path)
    
    if debug_mode:
        frames_save_path = os.path.join(args.save_latents_dir, f'frames')
        os.makedirs(frames_save_path, exist_ok=True)
        for i, rec_frame in enumerate(recon_frames):
            vis_frame = concat_images([depth_frames[i], frames[i], T.ToPILImage()(rec_frame)])
            vis_frame.save(os.path.join(frames_save_path, f'{i:03d}.png'))

def erase_latents(batch_latents, batch_erased_masks):
    rand_latents = torch.randn_like(batch_latents)
    batch_latents = torch.where(batch_erased_masks.unsqueeze(1).to(batch_latents.device), batch_latents, rand_latents)
    return batch_latents

def generate(args, model, prompt, negative_prompt, latents_save_path, img_size, depth_frames, fg_mask_frames, visibility_frames, selection, num_timesteps=50, guidance_scale=7.5):
    if len(selection) == 0:
        return []
    # set timesteps
    model.scheduler.set_timesteps(num_timesteps)

    # prepare foreground masks
    width, height = img_size
    seg_masks = prepare_fg_masks(fg_mask_frames, w=width, h=height, resolutions=[8, 16, 32])
    vis_masks = prepare_fg_masks(visibility_frames,  w=width, h=height, resolutions=[8, 16, 32])
    erased_masks = prepare_erase_masks(visibility_frames,  w=width, h=height)

    # register attention if PnP
    if args.is_pnp:
        fg_injection_t, bg_injection_t = init_method(model, pnp_attn_fg_t=args.pnp_attn_fg_t, pnp_attn_bg_t=args.pnp_attn_bg_t, conv_t=args.conv_t)
        print("fg_injection_t, bg_injection_t: ", fg_injection_t, bg_injection_t)
    else:
        fg_injection_t, bg_injection_t = 0, 0

    # prepare latents, cond
    depth_cond = prepare_depth_cond(depth_frames)
    windows = sliding_window(len(depth_frames), args.max_batch_size, args.overlap_size)
    print("sliding windows: ", windows)

    # prepare prompts
    inv_prompt = args.inversion_prompt

    # prepare text_embeds, added_cond_kwargs
    text_embeds, added_cond_kwargs = model.prepare_model_input(prompt, negative_prompt, inv_prompt, height, width, args.is_pnp)

    # sample
    for i, t in enumerate(tqdm(model.scheduler.timesteps, desc="Sampling")):
        geometric_latents = torch.load(latents_save_path)[i][selection]

        denoised_latents = torch.zeros_like(geometric_latents)
        counts = torch.zeros(len(denoised_latents)).to(model.device, model.dtype)

        for pivotal_idx in windows:
            # segmentation masks
            batch_seg_masks = {k: seg_masks[k][pivotal_idx].to(model.device) for k in seg_masks}
            batch_vis_masks = {k: vis_masks[k][pivotal_idx].to(model.device) for k in vis_masks}

            # prepare inputs and depth cond
            batch_denoised = erase_latents(geometric_latents[pivotal_idx], erased_masks[pivotal_idx]) if i == 0 else prev_denoised_latents[pivotal_idx]
            batch_depth_cond = depth_cond[pivotal_idx]

            # prepare source inputs
            source_latents = geometric_latents[pivotal_idx]

            # repeat text_embed_input and added_cond_kwargs
            input_text_embed = torch.repeat_interleave(text_embeds, len(batch_denoised), dim=0)
            input_added_kwargs = {k: torch.repeat_interleave(added_cond_kwargs[k], len(batch_denoised), dim=0) for k in added_cond_kwargs}
            
            # determine the injection mask to use
            if i <= min(bg_injection_t, fg_injection_t):
                input_batch_inj_masks = batch_vis_masks
            else:
                input_batch_inj_masks = batch_seg_masks

            # get batch attn masks
            input_batch_attn_masks = process_attn_mask(batch_seg_masks)

            # denoise
            batch_denoised = model.denoise_step(x=batch_denoised, 
                                                t=t,
                                                text_embeds=input_text_embed,
                                                added_cond_kwargs=input_added_kwargs,
                                                depth_cond=batch_depth_cond,
                                                guidance_scale=guidance_scale,
                                                is_pnp=args.is_pnp,
                                                source_latents=source_latents,
                                                batch_seg_masks=batch_seg_masks,
                                                batch_inj_masks=input_batch_inj_masks,
                                                batch_attn_masks=input_batch_attn_masks)

            # accumulate
            denoised_latents[pivotal_idx] += batch_denoised
            counts[pivotal_idx] += 1
        assert denoised_latents.shape[0] == counts.shape[0]
        prev_denoised_latents = denoised_latents / counts.view(-1,1,1,1)

    # decode and save
    decoded_frames = model.decode_latents(prev_denoised_latents, batch_size=args.vae_batch_size)
    decoded_frames = [T.ToPILImage()(frame) for frame in decoded_frames]

    return decoded_frames

if __name__ == "__main__":   
    args = parse_args()

    # get keyframes indexes
    num_frames = len([path for path in os.listdir(f"{args.data_path}/ani") if path[-3:]=="png"])
    keyframes = [i for i in range(1, num_frames, args.video_length-1)][:args.max_size]
 
    args.max_size = min(args.max_size, len(keyframes))

    # load keyframes frames with depths/masks
    image_open = lambda path: Image.open(path).convert("RGB")

    frames = [image_open(f"{args.data_path}/ani/{i:04d}.png") for i in keyframes]
    size = frames[0].size
        
    depth_frames = [normalize_depth(f"{args.data_path}/depth/{i:04d}.png", w=size[0], h=size[1]) for i in keyframes]
    fg_mask_frames = [Image.open(f"{args.data_path}/fg_mask/{i:04d}.png") for i in keyframes]
    visibility_frames = [Image.open(f"{args.data_path}/mask/{i:04d}.png") for i in keyframes]

    # create model
    model = ExtendAttnSD(args.lora_path)
    
    # inversion
    latents_save_path = os.path.join(args.save_latents_dir, f'latents.pt')
    if not os.path.exists(latents_save_path):
        args.skip_inversion = False
    if not args.skip_inversion:
        prep(args, model, depth_frames, frames, debug_mode=False)

    # 3d-guided keyframes generate
    gen_frames = generate(args, 
                       model,
                       prompt=args.prompt, 
                       negative_prompt="ugly, blurry, low res, unrealistic, unaesthetic", 
                       latents_save_path=latents_save_path, 
                       img_size=size, 
                       depth_frames = depth_frames, 
                       fg_mask_frames = fg_mask_frames, 
                       visibility_frames = visibility_frames, 
                       selection=torch.arange(0, args.max_size))

    os.makedirs(args.save_frames_dir, exist_ok=True)
    for frame, ki in zip(gen_frames, keyframes):
        i = (ki - 1) // (args.video_length - 1)
        save_path = os.path.join(args.save_frames_dir, f"{i:03d}.png")
        frame.save(save_path)

    # save the concate images
    concatenate_source(frames, gen_frames, f"{args.save_frames_dir}/debug.jpg")