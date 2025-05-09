import torch
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel

def pixelwise_correlation(real: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    """
    Compute pixel-wise Pearson correlation between ground truth and generated images.
    Args:
        real: Ground truth tensor of shape [N, C, H, W]
        generated: generated tensor of shape [N, C, H, W]
    Returns:
        corr_map: Tensor of shape [C, H, W] with correlation coefficients at each pixel
    """
    N, C, H, W = real.shape
    real_flat = real.view(N, C * H * W)
    generated_flat = generated.view(N, C * H * W)

    # Normalize across N (zero-mean and unit-variance)
    real_centered = real_flat - real_flat.mean(dim=0)
    generated_centered = generated_flat - generated_flat.mean(dim=0)

    numerator = (real_centered * generated_centered).sum(dim=0)
    denominator = torch.sqrt((real_centered ** 2).sum(dim=0) * (generated_centered ** 2).sum(dim=0) + 1e-8)

    corr = numerator / denominator
    return corr.view(C, H, W)



def pixelwise_corr_from_pil(real_imgs, generated_imgs):
    """
    Compute pixel-wise correlation from lists of PIL images.
    Args:
        real_imgs: list of ground truth PIL images
        generated_imgs: list of generated PIL images (same size/order)
    Returns:
        Tensor of shape [C, H, W] with per-pixel correlation values
    """
    assert len(real_imgs) == len(generated_imgs), "Mismatch in number of images"
    
    transform = T.ToTensor()  # Converts PIL to [C, H, W] float tensor (0–1)
    
    real_tensor = torch.stack([transform(img) for img in real_imgs])      # [N, C, H, W]
    generated_tensor = torch.stack([transform(img) for img in generated_imgs])  # [N, C, H, W]

    return pixelwise_correlation(real_tensor, generated_tensor)  # Uses function from earlier

def clip_difference(real_imgs:list[Image.Image], generated_imgs:list[Image.Image]):
    """
    Args:
        real_imgs: list of ground truth PIL images
        generated_imgs: list of generatedstructed PIL images (same size/order)
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    real_clip_list=[]
    for image in real_imgs:
        inputs = processor(images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs)[0]
        real_clip_list.append(image_features)

    generated_clip_list=[]
    for image in generated_imgs:
        inputs = processor(images=image, return_tensors="pt")
        image_features = model.get_image_features(**inputs)[0]
        generated_clip_list.append(image_features)

    return [F.mse_loss(real,generated).cpu().detach().item() for real,generated in zip(real_clip_list,generated_clip_list) ]