import torch
import torch
import torchvision.transforms as T

def pixelwise_correlation(gt: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """
    Compute pixel-wise Pearson correlation between ground truth and reconstructed images.
    Args:
        gt: Ground truth tensor of shape [N, C, H, W]
        recon: Reconstructed tensor of shape [N, C, H, W]
    Returns:
        corr_map: Tensor of shape [C, H, W] with correlation coefficients at each pixel
    """
    N, C, H, W = gt.shape
    gt_flat = gt.view(N, C * H * W)
    recon_flat = recon.view(N, C * H * W)

    # Normalize across N (zero-mean and unit-variance)
    gt_centered = gt_flat - gt_flat.mean(dim=0)
    recon_centered = recon_flat - recon_flat.mean(dim=0)

    numerator = (gt_centered * recon_centered).sum(dim=0)
    denominator = torch.sqrt((gt_centered ** 2).sum(dim=0) * (recon_centered ** 2).sum(dim=0) + 1e-8)

    corr = numerator / denominator
    return corr.view(C, H, W)



def pixelwise_corr_from_pil(gt_imgs, recon_imgs):
    """
    Compute pixel-wise correlation from lists of PIL images.
    Args:
        gt_imgs: list of ground truth PIL images
        recon_imgs: list of reconstructed PIL images (same size/order)
    Returns:
        Tensor of shape [C, H, W] with per-pixel correlation values
    """
    assert len(gt_imgs) == len(recon_imgs), "Mismatch in number of images"
    
    transform = T.ToTensor()  # Converts PIL to [C, H, W] float tensor (0–1)
    
    gt_tensor = torch.stack([transform(img) for img in gt_imgs])      # [N, C, H, W]
    recon_tensor = torch.stack([transform(img) for img in recon_imgs])  # [N, C, H, W]

    return pixelwise_correlation(gt_tensor, recon_tensor)  # Uses function from earlier