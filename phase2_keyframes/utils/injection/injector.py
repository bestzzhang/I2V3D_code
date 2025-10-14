from utils.injection.pnp_utils import (
    register_extended_attention_pnp, 
    register_conv_injection
)

def init_method(pipe, pnp_attn_fg_t=1.0, pnp_attn_bg_t=1.0, conv_t=1.0):
    timesteps = pipe.scheduler.timesteps
    n_timesteps = len(timesteps)
    print("num_inference_steps: ", n_timesteps)

    qk_fg_injection_t = int(n_timesteps * pnp_attn_fg_t)
    qk_bg_injection_t = int(n_timesteps * pnp_attn_bg_t)

    injection_t = max(qk_fg_injection_t, qk_bg_injection_t)
    injection_timesteps = timesteps[:injection_t] if injection_t >= 0 else []
    
    conv_injection_t = int(n_timesteps * conv_t)
    conv_injection_timesteps = timesteps[:conv_injection_t] if conv_injection_t >= 0 else []

    register_extended_attention_pnp(pipe, injection_timesteps)
    register_conv_injection(pipe, conv_injection_timesteps)
    return qk_fg_injection_t, qk_bg_injection_t
        
