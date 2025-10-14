from dataclasses import dataclass

import torch
from transformers import logging
from PIL import Image
# import PIL

from typing import List, Union
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from diffusers.image_processor import VaeImageProcessor
import numpy as np
from utils.models.injection.pnp_utils import register_time
from svd.scheduler.EDMinverse import EulerDiscreteInverseScheduler
from diffusers.schedulers import EulerDiscreteScheduler

logging.set_verbosity_error()

def tensor2vid(video: torch.Tensor, processor: VaeImageProcessor, output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil]")

    return outputs


class SVDInbetween():
    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def encode_vae_video(
        self,
        images,
        device: Union[str, torch.device],
        num_videos_per_prompt: int=1):
        
        do_classifier_free_guidance = False
        video_latents = []
        for image in images:
            width, height = image.size
            image = self.pipe.video_processor.preprocess(image, height=height, width=width).to(self.device)
            image_latents = self.pipe._encode_vae_image(
                image,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance
            )
            video_latents.append(image_latents)

        video_latents = torch.concat(video_latents, 0).unsqueeze(0)
        video_latents *= self.pipe.vae.config.scaling_factor
        return video_latents
    
    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device)
        image_latents = self.pipe.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _prepare_cond_inputs(self, controlnet_condition, height, width, repeats=1):
        controlnet_condition = self.pipe.video_processor.preprocess(controlnet_condition, height=height, width=width)
        controlnet_condition = controlnet_condition.unsqueeze(0)
        controlnet_condition = torch.repeat_interleave(controlnet_condition, repeats=repeats, dim=0)
        controlnet_condition = controlnet_condition.to(self.pipe.device, self.pipe.dtype)
        return controlnet_condition
    
    def _prepare_unet_inputs(self, 
                            generator,
                            first_frame,
                            num_frames,
                            fps,
                            height=320,
                            width=512,
                            motion_bucket_id=127,
                            noise_aug_strength=0.02,
                            batch_size=1,
                            num_videos_per_prompt=1,
                            do_classifier_free_guidance=True):
        ### 3. Encode image embedding
        image_embeddings = self.pipe._encode_image(first_frame, self.device, num_videos_per_prompt=num_videos_per_prompt, 
                                            do_classifier_free_guidance=do_classifier_free_guidance)
        
        ### 4. Encode image latents
        image = self.pipe.video_processor.preprocess(first_frame, height=height, width=width).to(self.device, dtype=self.pipe.dtype)
        noise = randn_tensor(image.shape, generator=generator, device=self.device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        image_latents = self.pipe._encode_vae_image(
                image,
                device=self.device,
                num_videos_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
        image_latents = image_latents.to(image_embeddings.dtype)
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
        ### 5. Get Added Time IDs
        fps = fps - 1
        added_time_ids = self.pipe._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(self.device)
            
        return image_embeddings, image_latents, added_time_ids
    
    def pred_noise(self,
                   latent_model_input,
                   t,
                   image_embeddings,
                   added_time_ids,
                   conditioning_scale=1.0,
                   depth_frames=None):
        if depth_frames != None:
            down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                controlnet_cond=depth_frames,
                added_time_ids=added_time_ids,
                conditioning_scale=conditioning_scale,
                guess_mode=False,
                return_dict=False,
            )
        else:
            down_block_res_samples=None
            mid_block_res_sample=None

        noise_pred = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=image_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]

        return noise_pred

    @torch.no_grad()
    def _sample(
            self,
            added_time_ids,
            guidance_scale: float,
            image_embeddings: torch.Tensor, 
            latents: torch.Tensor, 
            image_latents: torch.Tensor, 
            num_frames: int,
            depth_frames: List[Image.Image] = None,
            is_inversion: bool = False,
        ):
        decode_chunk_size = 1
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
        
        if is_inversion:
            # prepare the sigmas and timesteps
            sigmas = self.scheduler.set_inverse_sigmas()
            sigmas_0 = sigmas[0]
            init_t = torch.Tensor([0.25 * sigmas_0.log()]).to(self.device)
            timesteps = [init_t] + list(timesteps[1:])[::-1]
            # initialize latents list
            latents_list = [latents]

        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                     
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # predict noise
                noise_pred = self.pred_noise(
                    latent_model_input,
                    t,
                    image_embeddings=image_embeddings,
                    added_time_ids=added_time_ids,
                    conditioning_scale=1.0,
                    depth_frames=depth_frames,
                )
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                # reshape latents back
                if is_inversion:
                    latents_list.append(latents)
                    
                progress_bar.update()
        
        # scale the z0
        if is_inversion:
            latents_list[-1] = self.scheduler.prepare_last_latents(latents_list[-1])
            return latents_list
        
        video_tensor = self.pipe.decode_latents(latents, num_frames=num_frames, decode_chunk_size=decode_chunk_size)
        frames = tensor2vid(video_tensor, self.pipe.video_processor, output_type="pil")
        return frames[0]
    
    @torch.no_grad()
    def _interp_sample(
            self,
            added_time_ids,
            guidance_scale: float,
            image_embeddings: torch.Tensor, 
            latents: torch.Tensor, 
            image_latents: torch.Tensor, 
            num_frames: int,
            depth_frames: List[Image.Image] = None,
        ):
        decode_chunk_size = 1
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
        
        first_image_embeddings, end_image_embeddings = image_embeddings[0], image_embeddings[1]
        first_image_latents, end_image_latents = image_latents[0], image_latents[1]

        weights = torch.arange(num_frames)/(num_frames-1)
        weights = weights.view(1, num_frames, 1, 1, 1).to(self.device, end_image_embeddings.dtype)

        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict noise
                noise_pred = self.pred_noise(
                    torch.cat([latent_model_input, first_image_latents], dim=2),
                    t,
                    image_embeddings=first_image_embeddings,
                    added_time_ids=added_time_ids,
                    conditioning_scale=1.0,
                    depth_frames=depth_frames,
                )

                # reverse forward
                reverse_input = torch.flip(latent_model_input, dims=[1])
                reverse_noise_pred = self.pred_noise(
                    torch.cat([reverse_input, end_image_latents], dim=2),
                    t,
                    image_embeddings=end_image_embeddings,
                    added_time_ids=added_time_ids,
                    conditioning_scale=1.0,
                    depth_frames=torch.flip(depth_frames, dims=[1]) if depth_frames is not None else None
                )
                
                reverse_noise_pred = torch.flip(reverse_noise_pred, dims=[1]) # shape: (2, 14, 4, 64, 64)

                # use weights average
                noise_pred = reverse_noise_pred * weights + noise_pred * (1 - weights)
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                progress_bar.update()

        
        video_tensor = self.pipe.decode_latents(latents, num_frames=num_frames, decode_chunk_size=decode_chunk_size)
        frames = tensor2vid(video_tensor, self.pipe.video_processor, output_type="pil")
        return frames[0]
  
    @torch.no_grad()
    def _pnp_sample(
            self,
            added_time_ids: List[torch.Tensor],
            guidance_scale: float,
            image_embeddings: List[torch.Tensor], 
            latents: torch.Tensor, 
            latents_inv: List[torch.Tensor],
            image_latents: List[torch.Tensor],
            num_frames: int,
            depth_frames: List[torch.Tensor],
            interp_strength: float = 1.0,
            conditioning_scale: float = 0.5,
        ):
        decode_chunk_size = 1
        timesteps = self.scheduler.timesteps
        num_inference_steps = len(timesteps)
        
        first_image_embeddings = image_embeddings[0]
        first_image_latents = image_latents[0]
        first_latents_inv = latents_inv[0]
        if len(image_embeddings) == 2:
            end_image_embeddings = image_embeddings[1]
            end_image_latents = image_latents[1]
            end_latents_inv = latents_inv[1]
            weights = torch.arange(num_frames)/(num_frames-1)
            weights = weights.view(1, num_frames, 1, 1, 1).to(self.device, end_image_embeddings.dtype)
            is_inbetween = True
        else:
            is_inbetween = False

        interp_end = int(num_inference_steps * interp_strength)
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                register_time(self.pipe, t.item())
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                
                latent_model_input_concat = torch.cat([first_latents_inv[i], latent_model_input], 0)
                latent_model_input_concat = self.scheduler.scale_model_input(latent_model_input_concat, t)
                

                noise_pred = self.pred_noise(
                    torch.cat([latent_model_input_concat, first_image_latents], dim=2),
                    t,
                    image_embeddings=first_image_embeddings,
                    added_time_ids=added_time_ids,
                    conditioning_scale=conditioning_scale,
                    depth_frames=depth_frames
                )
                
                if is_inbetween and i <= interp_end:
                    # reverse forward
                    reverse_input = torch.flip(latent_model_input, dims=[1])
                    latent_model_input_concat = torch.cat([end_latents_inv[i], reverse_input], 0)
                    latent_model_input_concat = self.scheduler.scale_model_input(latent_model_input_concat, t)
        
                    reverse_noise_pred = self.pred_noise(
                        torch.cat([latent_model_input_concat, end_image_latents], dim=2),
                        t,
                        image_embeddings=end_image_embeddings,
                        added_time_ids=added_time_ids,
                        conditioning_scale=conditioning_scale,
                        depth_frames=torch.flip(depth_frames, dims=[1]) if depth_frames is not None else None
                    )
                    
                    reverse_noise_pred = torch.flip(reverse_noise_pred, dims=[1]) # shape: (2, 14, 4, 64, 64)
                    # use weights average
                    noise_pred = reverse_noise_pred * weights + noise_pred * (1 - weights)
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    _, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
 
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    
                progress_bar.update()
                
        video_tensor = self.pipe.decode_latents(latents, num_frames=num_frames, decode_chunk_size=decode_chunk_size)
        frames = tensor2vid(video_tensor, self.pipe.video_processor, output_type="pil")
        return frames[0]
    
    @torch.no_grad()
    def __call__(
            self,
            pipe, 
            first_frame: Image.Image,
            end_frame: Image.Image = None,
            depth_frames: List[Image.Image] = None,
            latents: torch.Tensor = None,
            num_videos_per_prompt: int = 1,
            height: int = 512,
            width: int = 320,
            num_frames: int = 14,
            guidance_scale: float = 1.5,
            num_inference_steps: int = 25,
            target_fps: int = 7,
            seed: int = 888
        ) -> Float[Tensor, "T B 4 H W"]:
        generator = torch.Generator().manual_seed(seed)

        self.pipe = pipe
        self.scheduler = self.pipe.scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-video-diffusion-img2vid",
            subfolder="scheduler",
        )
        
        self.device = self.pipe._execution_device
        self.pipe._guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1

        # 1. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 2. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            num_videos_per_prompt,
            num_frames=num_frames,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=pipe.dtype,
            device=self.device,
            generator=generator,
            latents=latents
        )

        # 3. Prepare unet inputs
        first_image_embeddings, fisrt_image_latents, added_time_ids = \
            self._prepare_unet_inputs(generator, first_frame, num_frames, target_fps, do_classifier_free_guidance=self.do_classifier_free_guidance, height=height, width=width)
        
        if end_frame:
            end_image_embeddings, end_image_latents, _ = \
                self._prepare_unet_inputs(generator, end_frame, num_frames, target_fps, do_classifier_free_guidance=self.do_classifier_free_guidance, height=height, width=width)

        # 4. Prepare depth inputs
        if depth_frames:
            repeats = 2 if self.do_classifier_free_guidance else 1
            depth_frames = self._prepare_cond_inputs(depth_frames, height=height, width=width, repeats=repeats)

        if end_frame is None:
            out = self._sample(
                added_time_ids,
                guidance_scale=guidance_scale,
                image_embeddings=first_image_embeddings,
                latents=latents,
                image_latents=fisrt_image_latents,
                num_frames=num_frames,
                depth_frames=depth_frames,
                is_inversion=False
            )
        else:
            out = self._interp_sample(
                added_time_ids,
                guidance_scale=guidance_scale,
                image_embeddings=[first_image_embeddings, end_image_embeddings],
                latents=latents,
                image_latents=[fisrt_image_latents, end_image_latents],
                num_frames=num_frames,
                depth_frames=depth_frames
            )
        return out

    @torch.no_grad()
    def sample_with_pnp(
            self,
            pipe, 
            first_frame: Image.Image = None,
            end_frame: Image.Image = None,
            src_images: List[Image.Image] = None,
            depth_frames: List[Image.Image] = None,
            start_code: torch.Tensor = None,
            first_latents_inv: torch.Tensor = None,
            end_latents_inv: torch.Tensor = None,
            height: int = 512,
            width: int = 320,
            guidance_scale: float = 1.5,
            interp_strength: float = 1.0,
            num_inference_steps: int = 25,
            target_fps: int = 8,
            conditioning_scale: float = 1.0,
            seed: int = 888
        ) -> Float[Tensor, "T B 4 H W"]:
        generator = torch.Generator().manual_seed(seed)
        num_frames = start_code.shape[1]


        self.pipe = pipe
        self.scheduler = self.pipe.scheduler
            
        self.device = self.pipe._execution_device
        self.pipe._guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1

        # 1. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 2. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        
        dtype = self.pipe.dtype
        start_code = self.pipe.prepare_latents(
            1,
            num_frames=num_frames,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=self.device,
            generator=generator,
            latents=start_code
        )

        # 3. Prepare unet inputs
        image_embeddings, image_latents, added_time_ids = \
            self._prepare_unet_inputs(generator, first_frame, num_frames, target_fps, do_classifier_free_guidance=True, height=height, width=width)
        
        ddim_image_embeddings, ddim_image_latents, ddim_added_time_ids = \
            self._prepare_unet_inputs(generator, src_images[0], num_frames, target_fps,  do_classifier_free_guidance=False, height=height, width=width)

        image_embeddings = torch.cat([ddim_image_embeddings, image_embeddings])
        image_latents = torch.cat([ddim_image_latents, image_latents])
        added_time_ids = torch.cat([ddim_added_time_ids, added_time_ids])
        
        
        if end_frame != None:
            reverse_image_embeddings, reverse_image_latents, _ = \
                self._prepare_unet_inputs(generator, end_frame, num_frames, target_fps, do_classifier_free_guidance=True, height=height, width=width)
        
            reverse_ddim_image_embeddings, reverse_ddim_image_latents, _ = \
                self._prepare_unet_inputs(generator, src_images[-1], num_frames, target_fps,  do_classifier_free_guidance=False, height=height, width=width)
            
            reverse_image_embeddings = torch.cat([reverse_ddim_image_embeddings, reverse_image_embeddings])
            reverse_image_latents = torch.cat([reverse_ddim_image_latents, reverse_image_latents])
        
        # 4. Prepare depth inputs
        if depth_frames:
            depth_frames = self._prepare_cond_inputs(depth_frames, height=height, width=width, repeats=3)

        if end_frame != None:
            out = self._pnp_sample(
                added_time_ids=added_time_ids,
                guidance_scale=guidance_scale,
                image_embeddings=[image_embeddings, reverse_image_embeddings],
                latents=start_code,
                latents_inv=[first_latents_inv, end_latents_inv],
                image_latents=[image_latents, reverse_image_latents],
                num_frames=num_frames,
                depth_frames=depth_frames,
                interp_strength = interp_strength,
                conditioning_scale=conditioning_scale
            )
        else:
            print("Not use inbetween")
            out = self._pnp_sample(
                added_time_ids=added_time_ids,
                guidance_scale=guidance_scale,
                image_embeddings=[image_embeddings],
                latents=start_code,
                latents_inv=[first_latents_inv],
                image_latents=[image_latents],
                num_frames=num_frames,
                depth_frames=depth_frames,
                conditioning_scale=conditioning_scale
            )
        return out
    
    @torch.no_grad()
    def get_inverted_latent(
            self,
            pipe, 
            src_images: List[Image.Image],
            depth_frames: List[Image.Image] = None,
            height: int = 512,
            width: int = 320,
            num_frames: int = 14,
            guidance_scale: float = 1.0,
            num_inference_steps: int = 25,
            target_fps: int = 8,
            seed: int = 888
        ) -> Float[Tensor, "T B 4 H W"]:
        generator = torch.Generator().manual_seed(seed)

        self.pipe = pipe
        self.scheduler = self.pipe.scheduler = EulerDiscreteInverseScheduler.from_pretrained("stabilityai/stable-video-diffusion-img2vid",
            subfolder="scheduler",
        )
            
        self.device = self.pipe._execution_device
        self.pipe._guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1

        # 1. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 2. Prepare latent variables
        latents = self.encode_vae_video(src_images, self.device).to(pipe.dtype) ### This is modification from __call__

        # 3. Prepare unet inputs
        first_frame = src_images[0]
        image_embeddings, image_latents, added_time_ids = \
            self._prepare_unet_inputs(generator, first_frame, num_frames, target_fps, do_classifier_free_guidance=self.do_classifier_free_guidance, height=height, width=width)
        
        # 4. Prepare depth inputs
        if depth_frames:
            repeats = 2 if self.do_classifier_free_guidance else 1
            depth_frames = self._prepare_cond_inputs(depth_frames, height=height, width=width, repeats=repeats)

        out = self._sample(
            added_time_ids,
            guidance_scale=guidance_scale,
            image_embeddings=image_embeddings,
            latents=latents,
            image_latents=image_latents,
            num_frames=num_frames,
            depth_frames=depth_frames,
            is_inversion=True
        )
        return out

    
