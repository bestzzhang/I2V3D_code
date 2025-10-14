import torch
import random
import numpy as np
from PIL import Image
import os
import yaml
import torch.nn.functional as F
import cv2

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def sliding_window(n, window_size=4, overlaps=1):
    windows = []
    step = window_size - overlaps
    for i in range(0, n, step):
        window = torch.arange(i, min(i + window_size, n))
        windows.append(window)
    if len(windows)>1 and len(windows[-1]) == overlaps:
        windows.pop()
    return windows

def add_dict_to_yaml_file(file_path, key, value):
    data = {}

    # If the file already exists, load its contents into the data dictionary
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the data back to the YAML file
    with open(file_path, 'w') as file:
        yaml.dump(data, file)
        
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def concat_images(images, target_width=512, target_height=288, max_per_row=4):
    # Load all images and resize them to the target width and height
    img_list = [img.resize((target_width, target_height)) for img in images]
    
    # Calculate number of rows and columns
    num_images = len(images)
    num_rows = (num_images + max_per_row - 1) // max_per_row
    num_cols = min(num_images, max_per_row)
    
    # Create a blank canvas for the final concatenated image
    concat_width = target_width * num_cols
    concat_height = target_height * num_rows
    concat_image = Image.new('RGB', (concat_width, concat_height))
    
    # Place each resized image on the canvas
    for idx, img in enumerate(img_list):
        row, col = divmod(idx, max_per_row)
        x_offset = col * target_width
        y_offset = row * target_height
        concat_image.paste(img, (x_offset, y_offset))
    
    return concat_image

def prepare_depth_cond(depth_frames):
    depth_maps = []
    for dep_frame in depth_frames:
        image = np.array(dep_frame)
        image = torch.from_numpy((image.astype(np.float32) / 255.0))
        depth_maps.append(image)
        
    depth_maps = torch.stack(depth_maps).permute(0, 3, 1, 2)
    return depth_maps

def concatenate_source(src_frames, edit_frames, output_path):
    # Determine the size of the final combined image
    combined_width = 2 * src_frames[0].width
    combined_height = len(src_frames) * src_frames[0].height
    final_image = Image.new("RGB", (combined_width, combined_height))

    # Iterate and concatenate each pair of frames
    y_offset = 0
    for src, edit in zip(src_frames, edit_frames):
        combined_frame = Image.new("RGB", (combined_width, src.height))
        combined_frame.paste(src, (0, 0))
        combined_frame.paste(edit, (src.width, 0))
        
        # Paste the combined frame into the final image
        final_image.paste(combined_frame, (0, y_offset))
        y_offset += src.height

    # Save the final image
    final_image.save(output_path)

def prepare_fg_masks(masks, w, h, use_flatten=True, resolutions=[16, 32]):
    downscales = []
    for resol in resolutions:
        res_h, res_w = h // resol, w // resol
        downscales.append((res_h, res_w))
        
    masks = torch.from_numpy(np.array(masks)==255).float()
    # print("mask shape: ", masks.shape)
    
    seg_masks = {}
    for (th, tw) in downscales:
        ## resizing using interpolate
        sequence_length = th * tw
        resized_masks = F.interpolate(masks.unsqueeze(1), size=(th, tw), mode='nearest').squeeze(1).bool()
        resized_masks = resized_masks.view(len(masks), sequence_length) if use_flatten else resized_masks
        seg_masks[sequence_length] = resized_masks
        
    return seg_masks

def prepare_erase_masks(masks, w, h, resol=8):
    masks = torch.from_numpy(np.array(masks)==255).float()
    res_h, res_w = h // resol, w // resol
    resized_masks = F.interpolate(masks.unsqueeze(1), size=(res_h, res_w), mode='nearest').squeeze(1).bool()
    # print("erased mask shape: ", resized_masks.shape)
    return resized_masks

def process_attn_mask(batch_seg_masks):
    ret = {}
    for k in batch_seg_masks:
        m = batch_seg_masks[k]
        num_frames, seq_len = m.shape
        m_mask = m.unsqueeze(-1) # (num_frames, seq_len, 1)
        m_expanded = m.view(-1) # (num_frames * seq_len)
        
        # batch_attn_mask = torch.zeros((num_frames, seq_len, num_frames * seq_len), dtype=torch.bool).to("cuda")
        batch_attn_mask = torch.where(m_mask, False, ~m_expanded)
        for i in range(num_frames):
            batch_attn_mask[i, :, i*seq_len:(i+1)*seq_len] = True
        batch_attn_mask = batch_attn_mask.unsqueeze(0).repeat(2, 1, 1, 1)
        batch_attn_mask = batch_attn_mask.view(-1, seq_len, seq_len * num_frames).unsqueeze(1)
        ret[k] = batch_attn_mask
    return ret
    
    

def normalize_depth(image_path, w, h, output_path=None):
    """
    Extracts the R channel from an RGB image, normalizes it to 0-255,
    and duplicates it into an RGB format.
    
    Args:
        image_path (str): Path to the input RGB image.
        output_path (str, optional): Path to save the output image. If None, the image is not saved.
    
    Returns:
        np.ndarray: The resulting RGB image where all channels are the normalized R channel.
    """
    # Load the RGB image
    rgb_image = cv2.imread(image_path)
    
    if rgb_image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    rgb_image = cv2.resize(rgb_image, (w, h))
    
    # Extract the R channel (OpenCV uses BGR by default)
    r_channel = rgb_image[:, :, 2]
    
    # Normalize the R channel to the range 0-255
    r_channel_normalized = cv2.normalize(r_channel, None, 0, 255, cv2.NORM_MINMAX)
    
    # Duplicate the normalized R channel into RGB format
    rgb_from_r = cv2.merge([r_channel_normalized, r_channel_normalized, r_channel_normalized])
    
    # Save the output image if a path is provided
    if output_path:
        cv2.imwrite(output_path, rgb_from_r)
    
    return Image.fromarray(rgb_from_r)


def frames_to_video(frames, output_video_path, fps=8):
    from torchvision.io import write_video
    new_frames = []

    for image in frames:
        frame = torch.from_numpy(np.array(image))
        new_frames.append(frame)

    frames_tensor = torch.stack(new_frames)
    
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }

    write_video(output_video_path, frames_tensor, fps=fps, video_codec=video_codec, options=video_options)

