from utils.models.injection.pnp_utils import (
    register_conv_injection,
    register_spatial_attention_pnp
)



def init_pnp(pipe, pnp_f_t, pnp_spatial_attn_t, num_inference_steps=25):
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    num_inference_steps = len(timesteps)
    
    print("num_inference_steps: ", num_inference_steps)
    
    conv_injection_t = int(num_inference_steps * pnp_f_t)
    spatial_attn_qk_injection_t = int(num_inference_steps * pnp_spatial_attn_t)
    
    conv_injection_timesteps = (
        timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
    )
    
    spatial_attn_qk_injection_timesteps = (
        timesteps[:spatial_attn_qk_injection_t] if spatial_attn_qk_injection_t >= 0 else []
    )

    register_conv_injection(pipe, conv_injection_timesteps)
    register_spatial_attention_pnp(pipe, spatial_attn_qk_injection_timesteps)

