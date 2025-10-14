
import torch
import torch.nn.functional as F

def register_time(model, t):
    up_res_dict = {
            0: {0: list(range(10)), 1: list(range(10)), 2: list(range(10))},
            1: {0: list(range(2)), 1: list(range(2)), 2: list(range(2))}
    }
    
    for res in up_res_dict:
        for block in up_res_dict[res]:
            for tf_block in up_res_dict[res][block]:
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[tf_block].attn1
                setattr(module, 't', t)
    
    for j in range(3):
        for i in range(3):
            conv_module = model.unet.up_blocks[j].resnets[i]
            setattr(conv_module, 't', t)

def register_inject_masks(model, batch_inject_masks):
    up_res_dict = {
            0: {0: list(range(10)), 1: list(range(10)), 2: list(range(10))},
            1: {0: list(range(2)), 1: list(range(2)), 2: list(range(2))}
    }
    
    for res in up_res_dict:
        for block in up_res_dict[res]:
            for tf_block in up_res_dict[res][block]:
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[tf_block].attn1
                setattr(module, 'batch_inject_masks', batch_inject_masks)

def register_seg_masks(model, batch_seg_masks):    
    # TODO: change the reshape_masks to be not hard coded
    if batch_seg_masks != None:
        batch_size = batch_seg_masks[576].shape[0]
        reshape_masks = {}
        reshape_masks[576] = batch_seg_masks[576].view(batch_size, 18, 32)
        reshape_masks[2304] = batch_seg_masks[2304].view(batch_size, 36, 64)
        reshape_masks[9216] = batch_seg_masks[9216].view(batch_size, 72, 128)
    else:
        reshape_masks = None

    for j in range(3):
        for i in range(3):
            module = model.unet.up_blocks[j].resnets[i]
            setattr(module, 'batch_seg_masks', reshape_masks)

def register_attn_masks(model, batch_attn_masks):
    up_res_dict = {
            0: {0: list(range(10)), 1: list(range(10)), 2: list(range(10))},
            1: {0: list(range(2)), 1: list(range(2)), 2: list(range(2))}
    }
    
    for res in up_res_dict:
        for block in up_res_dict[res]:
            for tf_block in up_res_dict[res][block]:
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[tf_block].attn1
                setattr(module, 'batch_attn_masks', batch_attn_masks)

def register_conv_injection(model, injection_schedule, use_image_sculpting=False):
    def conv_forward(self):
        def forward(input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule):
                source_batch_size = int(hidden_states.shape[0] // 3)
                if self.batch_seg_masks != None:
                    # reshape
                    *_, res_h, res_w = hidden_states.shape
                    seq_len = int(res_h * res_w)
                    seg_masks = self.batch_seg_masks[seq_len].unsqueeze(1)
                    # inject unconditional
                    hidden_states[source_batch_size:2 * source_batch_size] = torch.where(seg_masks, hidden_states[:source_batch_size], hidden_states[source_batch_size:2 * source_batch_size])
                    # inject conditional
                    hidden_states[2 * source_batch_size:] = torch.where(seg_masks, hidden_states[:source_batch_size], hidden_states[2 * source_batch_size:])
                else:
                    # inject unconditional
                    hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                    # inject conditional
                    hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    num_layers = 1 if use_image_sculpting else 3
    for j in range(num_layers):
        for i in range(3):
            conv_module = model.unet.up_blocks[j].resnets[i]
            conv_module.forward = conv_forward(conv_module)
            setattr(conv_module, 'injection_schedule', injection_schedule)

def register_extended_attention_pnp(model, injection_timesteps):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            query = self.to_q(x)
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            # injection
            if self.injection_schedule is not None and self.t in self.injection_schedule:
                if self.batch_inject_masks != None:
                    # inject unconditional
                    batch_inject_masks = self.batch_inject_masks[sequence_length].unsqueeze(-1)
                    query[n_frames:2 * n_frames] = torch.where(batch_inject_masks, query[:n_frames], query[n_frames:2 * n_frames])
                    key[n_frames:2 * n_frames] = torch.where(batch_inject_masks, key[:n_frames], key[n_frames:2 * n_frames])
                    # inject conditional
                    query[2 * n_frames:] = torch.where(batch_inject_masks, query[:n_frames], query[2 * n_frames:])
                    key[2 * n_frames:] = torch.where(batch_inject_masks, key[:n_frames], key[2 * n_frames:])
                else:
                    # inject unconditional
                    query[n_frames:2 * n_frames] = query[:n_frames]
                    key[n_frames:2 * n_frames] = key[:n_frames]
                    # inject conditional
                    query[2 * n_frames:] = query[:n_frames]
                    key[2 * n_frames:] = key[:n_frames]
            

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            # process K, Q, and V
            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            q_source = query[:n_frames]
            q_target = query[n_frames:]

            k_source = key[:n_frames].view(n_frames, -1, self.heads, head_dim).transpose(1, 2)
            k_target = key[n_frames:]
            k_target = k_target.view(2, 1, n_frames * sequence_length, self.heads, head_dim)
            k_target = k_target.repeat(1, n_frames, 1, 1, 1).view(2 * n_frames, n_frames * sequence_length, self.heads, head_dim)
            k_target = k_target.permute(0, 2, 1, 3).contiguous()
            
            v_source = value[:n_frames].view(n_frames, -1, self.heads, head_dim).transpose(1, 2)
            v_target = value[n_frames:]
            v_target = v_target.view(2, 1, n_frames * sequence_length, self.heads, head_dim)
            v_target = v_target.repeat(1, n_frames, 1, 1, 1).view(2 * n_frames, n_frames * sequence_length, self.heads, head_dim)
            v_target = v_target.permute(0, 2, 1, 3).contiguous()

            if self.batch_attn_masks != None:
                batch_attn_masks = self.batch_attn_masks[sequence_length]  

            # out computation
            out_source = F.scaled_dot_product_attention(
                q_source, k_source, v_source, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            out_target = F.scaled_dot_product_attention(
                q_target, k_target, v_target, attn_mask=batch_attn_masks, dropout_p=0.0, is_causal=False
            )

            # to_out
            encoder_hidden_states = torch.cat([out_source, out_target], dim=0)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            encoder_hidden_states = encoder_hidden_states.to(query.dtype)

            return to_out(encoder_hidden_states)

        return forward

    res_dict = {
            0: {0: list(range(10)), 1: list(range(10)), 2: list(range(10))},
            1: {0: list(range(2)), 1: list(range(2)), 2: list(range(2))}
    }
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            for tf_block in res_dict[res][block]:
                module = model.unet.up_blocks[res].attentions[block].transformer_blocks[tf_block].attn1
                module.forward = sa_forward(module)
                setattr(module, 'injection_schedule', injection_timesteps)
                setattr(module, 'batch_attn_masks', None)
