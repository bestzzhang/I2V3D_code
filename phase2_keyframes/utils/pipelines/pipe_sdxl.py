from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from utils.injection.pnp_utils import register_time, register_inject_masks, register_seg_masks, register_attn_masks
# suppress partial model loading warning
logging.set_verbosity_error()

from tqdm import tqdm
import torch
import torchvision.transforms as T
import numpy as np

class ExtendAttnSD():
    def __init__(self, lora_path):
        super().__init__()

        self.device = "cuda"
        self.dtype = torch.float16

        print(f'[INFO] loading stable diffusion...')

        # Create model
        model_key = "stabilityai/stable-diffusion-xl-base-1.0"
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0",
            torch_dtype=torch.float16
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        control_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_key, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
        ).to(self.device)
        control_pipe.load_lora_weights(lora_path)

        self.pipe = control_pipe
        self.unet = control_pipe.unet
        self.controlnet = control_pipe.controlnet
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        
        # self.unet.enable_xformers_memory_efficient_attention()
        print(f'[INFO] loaded stable diffusion!')
    
    @torch.no_grad()
    def prepare_model_input(self, prompt, negative_prompt, inv_prompt, height, width, is_pnp=False):
        #  Text embeddings
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )

        if is_pnp:
            (
                inv_prompt_embeds,
                _,
                inv_pooled_prompt_embeds,
                _
            ) = self.pipe.encode_prompt(
                prompt=inv_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        
        text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        add_time_ids = self.pipe._get_add_time_ids(
            (height, width), (0, 0), (height, width), dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim
        )
        
        if is_pnp:
            add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids], dim=0)
            add_text_embeds = torch.cat([inv_pooled_prompt_embeds, negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            text_embeds = torch.cat([inv_prompt_embeds, negative_prompt_embeds, prompt_embeds], dim=0).to(self.device)
        else:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            text_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(self.device)

        added_cond_kwargs = {"text_embeds": add_text_embeds.to(self.device), "time_ids": add_time_ids.to(self.device)}
        return text_embeds, added_cond_kwargs

    @torch.no_grad()
    def prepare_depth_maps(self, depth_frames):
        depth_maps = []
        for frame in depth_frames:
            image = np.array(frame)
            image = torch.from_numpy((image.astype(np.float32) / 255.0))
            depth_maps.append(image)
        depth_maps = torch.stack(depth_maps).permute(0, 3, 1, 2).to(self.device).to(torch.float16)
        return depth_maps

    def controlnet_pred(self, latent_model_input, t, text_embed_input, controlnet_cond, added_cond_kwargs):
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embed_input,
                controlnet_cond=controlnet_cond,
                conditioning_scale=1,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
        
        # apply the denoising network
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            cross_attention_kwargs={},
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return noise_pred
    
    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def decode_latents(self, latents, batch_size):
        decoded = []
        for b in range(0, latents.shape[0], batch_size):
                latents_batch = 1 / self.pipe.vae.config.scaling_factor * latents[b:b + batch_size]
                imgs = self.pipe.vae.decode(latents_batch).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1)
                decoded.append(imgs)
        return torch.cat(decoded)

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def encode_imgs(self, imgs, batch_size, deterministic=True):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.pipe.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * self.pipe.vae.config.scaling_factor)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def encode_PIL(self, frame):
        frame = T.ToTensor()(frame).to(self.device).unsqueeze(0)
        frame = 2 * frame - 1
        posterior = self.pipe.vae.encode(frame).latent_dist
        latents = posterior.sample() * self.pipe.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, depth_cond, added_cond_kwargs, latent_frames, batch_size):
        timesteps = reversed(self.scheduler.timesteps)
        save_latent_frames = []
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, latent_frames.shape[0], batch_size):
                x_batch = latent_frames[b:b + batch_size]
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
                                                                    
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.controlnet_pred(x_batch, t, cond_batch, torch.cat([depth_cond[b: b + batch_size]]), added_cond_kwargs=added_cond_kwargs)
                
                pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
                latent_frames[b:b + batch_size] = mu * pred_x0 + sigma * eps

            save_latent_frames.append(latent_frames.clone())
        return save_latent_frames

    @torch.no_grad()
    def ddim_sample(self, x, cond, depth_cond, added_cond_kwargs, batch_size=4):
        timesteps = self.scheduler.timesteps
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, x.shape[0], batch_size):
                x_batch = x[b:b + batch_size]
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
                
                
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.controlnet_pred(x_batch, t, cond_batch, torch.cat([depth_cond[b: b + batch_size]]), added_cond_kwargs=added_cond_kwargs)

                pred_x0 = (x_batch - sigma * eps) / mu
                x[b:b + batch_size] = mu_prev * pred_x0 + sigma_prev * eps
        rgb_reconstruction = self.decode_latents(x, batch_size=batch_size)
        return rgb_reconstruction

    @torch.no_grad()
    def extract_latents(self, 
                        frames,
                        depth_frames,
                        batch_size=4,
                        num_steps=50,
                        inversion_prompt='',
                        debug_mode=False):
        # process frames
        depth_cond = self.prepare_depth_maps(depth_frames)
        
        # process depth frames
        width, height = frames[0].size
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(self.device, dtype=self.dtype)
        latent_frames = self.encode_imgs(frames, batch_size=batch_size, deterministic=True).to(self.device, dtype=torch.float16)

        # prepare prompts        
        (
            inv_prompt_embeds,
            _,
            inv_pooled_prompt_embeds,
            _
        ) = self.pipe.encode_prompt(
            prompt=inversion_prompt,
            device=self.device,
            do_classifier_free_guidance=False,
        )
        
        # set timesteps
        self.scheduler.set_timesteps(num_steps)

        # prepare added_cond_kwargs
        add_time_ids = self.pipe._get_add_time_ids(
            (height, width), (0, 0), (height, width), dtype=inv_prompt_embeds.dtype, 
            text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim
        ).to(self.device)

        added_cond_kwargs = {
            "text_embeds": inv_pooled_prompt_embeds, 
            "time_ids": add_time_ids
        }
        
        # invert
        inverted_x = self.ddim_inversion(inv_prompt_embeds,
                                         depth_cond,
                                         added_cond_kwargs,
                                         latent_frames,
                                         batch_size=batch_size)
        
        if debug_mode:
            frames = self.ddim_sample(inverted_x[-1], inv_prompt_embeds, depth_cond, added_cond_kwargs, batch_size=batch_size)
            return frames, inverted_x
        return inverted_x

    @torch.no_grad()
    def denoise_step(self,
                     x, 
                     t,
                     text_embeds,
                     added_cond_kwargs,
                     depth_cond, 
                     is_pnp=False,
                     source_latents=None,
                     batch_seg_masks=None,
                     batch_inj_masks=None,
                     batch_attn_masks=None,
                     guidance_scale=7.5):
        if is_pnp:
            register_time(self, t.item())
            register_inject_masks(self, batch_inject_masks=batch_inj_masks)
            register_seg_masks(self, batch_seg_masks=batch_seg_masks)
            latent_model_input = torch.cat([source_latents] + ([x] * 2))
            cond_image = torch.cat([depth_cond] * 3).to(self.device, self.dtype)
        else:
            latent_model_input = torch.cat(([x] * 2))
            cond_image = torch.cat([depth_cond] * 2).to(self.device, self.dtype)
        
        if batch_attn_masks != None:
            register_attn_masks(self, batch_attn_masks=batch_attn_masks)
        
 
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeds,
                controlnet_cond=cond_image,
                conditioning_scale=1.0,
                guess_mode=False,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
        
        # apply the denoising network
        noise_pred = self.unet(latent_model_input, 
                               t, 
                               encoder_hidden_states=text_embeds, 
                               cross_attention_kwargs={},
                               down_block_additional_residuals=down_block_res_samples,
                               mid_block_additional_residual=mid_block_res_sample,
                               return_dict=False,
                               added_cond_kwargs=added_cond_kwargs)[0]

        # perform guidance
        if is_pnp:
            _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        else:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent
    
    