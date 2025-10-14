from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
from torchvision.io import write_video

def pil_to_tensor_with_batch(image):
    """
    Convert a PIL image to a tensor with a batch dimension.
    
    Parameters:
    image (PIL.Image): The image to convert.
    
    Returns:
    torch.Tensor: The converted tensor with a batch dimension.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL Image")

    # Define the transformation to convert the PIL image to a tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the PIL image to a tensor
        transforms.Lambda(lambda x: x.unsqueeze(0))  # Add a batch dimension
    ])
    
    # Apply the transformation
    tensor = transform(image)
    
    return tensor

def load_image(path, size=(512,512)):
    return Image.open(path).resize(size)

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

def prepare_masks(masks, resolutions=[8, 16, 32]):
    import torch.nn.functional as F
    width, height = masks[0].size
    
    downscales = []
    for resol in resolutions:
        res_h, res_w = height // resol, width // resol
        downscales.append((res_h, res_w))

    masks = torch.from_numpy(np.array(masks)==255).float()
    
    seg_masks = {}
    for (th, tw) in downscales:
        ## resizing using interpolate
        sequence_length = th * tw
        resized_masks = F.interpolate(masks.unsqueeze(1), size=(th, tw), mode='nearest').squeeze(1).bool()
        seg_masks[sequence_length] = resized_masks.to("cuda")
        
    return seg_masks

def frames_to_video(frames, output_video_path, fps=8):
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

def save_concat(source_frames, all_frames, concat_video_name, fps=24):
    """
    Vertically concatenate lists of PIL.Image frames and save the result as a video.
    Assumes both lists have equal length and each pair of images share the same width and height.
    """
    concat_frames = []
    for src, gen in zip(source_frames, all_frames):
        w, h = src.size
        concat_img = Image.new("RGB", (w, h * 2))
        concat_img.paste(src, (0, 0))
        concat_img.paste(gen, (0, h))
        concat_frames.append(concat_img)

    frames_to_video(concat_frames, concat_video_name, fps=fps)