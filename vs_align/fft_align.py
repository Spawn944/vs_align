"""
FFT Phase Correlation for Global Translation Alignment
Uses scipy.signal.correlate for robust sub-pixel shift estimation.
"""
import numpy as np
from scipy.signal import correlate
from scipy.ndimage import fourier_shift
import torch

def estimate_shift_fft(ref_np: np.ndarray, clip_np: np.ndarray, 
                       max_shift: int = 100, upsample_factor: int = 100) -> tuple:
    """
    Estimate global translation shift between two images using phase correlation.
    
    Args:
        ref_np: Reference frame (H, W) float32 [0, 1]
        clip_np: Clip frame (H, W) float32 [0, 1]
        max_shift: Maximum pixel shift to search for
        upsample_factor: Factor for sub-pixel refinement
        
    Returns:
        tuple: (shift_y, shift_x, confidence)
               If confidence < threshold, shift should be ignored.
    """
    if ref_np.shape != clip_np.shape:
        raise ValueError("Input shapes must match")
    
    h, w = ref_np.shape
    
    # Apply Hanning window to reduce edge effects
    wy = np.hanning(h)
    wx = np.hanning(w)
    window = wy[:, None] * wx[None, :]
    
    ref_win = ref_np * window
    clip_win = clip_np * window
    
    # Normalize
    ref_win -= ref_win.mean()
    clip_win -= clip_win.mean()
    
    # Compute cross-correlation via FFT
    # mode='full' gives output size 2*h-1, 2*w-1
    corr = correlate(ref_win, clip_win, mode='full', method='fft')
    
    # Find peak in correlation surface
    # Define search region around center
    cy, cx = h - 1, w - 1
    y_min, y_max = max(0, cy - max_shift), min(2*h - 1, cy + max_shift + 1)
    x_min, x_max = max(0, cx - max_shift), min(2*w - 1, cx + max_shift + 1)
    
    search_region = corr[y_min:y_max, x_min:x_max]
    
    if search_region.size == 0:
        return 0.0, 0.0, 0.0
        
    # Coarse peak location
    local_peak = np.unravel_index(np.argmax(search_region), search_region.shape)
    peak_y = local_peak[0] + y_min
    peak_x = local_peak[1] + x_min
    
    # Calculate coarse shift relative to center
    coarse_shift_y = peak_y - cy
    coarse_shift_x = peak_x - cx
    
    # Sub-pixel refinement using matrix-multiplication based upsampling
    # We only upsample a small region around the coarse peak for efficiency
    # Using scipy's fourier_shift approach or manual DFT upsampling
    # Here we implement a simplified version: upsample the region around the peak
    
    # For high precision, we can use the property that shifting in spatial domain
    # is multiplication by linear phase ramp in frequency domain.
    # However, a simpler robust approach for this specific use case:
    # Just return coarse if max_shift is small, or use a library function if available.
    # Since we want to avoid heavy dependencies, let's use a localized upsampling trick.
    
    # Refined calculation using Fourier upsampling of the correlation peak
    # This is computationally expensive if done on full image, so we do it locally?
    # Actually, scipy.signal.correlate doesn't return frequency domain.
    # Let's use a standard phase correlation implementation with upsampling.
    
    # Re-implementing using direct FFT for better control over upsampling
    f0 = np.fft.fft2(ref_win)
    f1 = np.fft.fft2(clip_win)
    
    # Cross-power spectrum
    r = f0 * np.conj(f1)
    denom = np.abs(f0) * np.abs(f1)
    # Avoid division by zero
    denom[denom < 1e-6] = 1e-6
    r /= denom
    
    # Inverse FFT to get correlation
    corr_full = np.fft.ifft2(r)
    corr_full = np.real(corr_full)
    corr_full = np.fft.fftshift(corr_full) # Shift zero-frequency to center
    
    # Peak location in shifted array
    cy, cx = h // 2, w // 2
    y_min, y_max = max(0, cy - max_shift), min(h, cy + max_shift + 1)
    x_min, x_max = max(0, cx - max_shift), min(w, cx + max_shift + 1)
    
    search_region = corr_full[y_min:y_max, x_min:x_max]
    if search_region.size == 0:
        return 0.0, 0.0, 0.0
        
    local_peak = np.unravel_index(np.argmax(search_region), search_region.shape)
    peak_y = local_peak[0] + y_min
    peak_x = local_peak[1] + x_min
    
    coarse_shift_y = peak_y - cy
    coarse_shift_x = peak_x - cx
    
    # Sub-pixel refinement via matrix-multiplication DFT upsampling
    # Only upsample a 1x1 region around the peak to high precision
    # This is equivalent to zooming into the correlation peak
    
    # Parameters for upsampling
    upsampled_region_size = 1 # We want the peak location specifically
    
    # Using a simplified approach: 
    # If we need sub-pixel, we can fit a paraboloid to the 3x3 neighborhood
    # This is often fast and accurate enough for video alignment
    if 0 < peak_y < h-1 and 0 < peak_x < w-1:
        neighborhood = corr_full[peak_y-1:peak_y+2, peak_x-1:peak_x+2]
        if neighborhood.shape == (3, 3):
            # Parabolic fit
            # dx = 0.5 * (val_left - val_right) / (2*val_center - val_left - val_right)
            center = neighborhood[1, 1]
            left = neighborhood[1, 0]
            right = neighborhood[1, 2]
            top = neighborhood[0, 1]
            bottom = neighborhood[2, 1]
            
            denom_x = 2 * center - left - right
            denom_y = 2 * center - top - bottom
            
            if abs(denom_x) > 1e-6:
                sub_x = 0.5 * (left - right) / denom_x
            else:
                sub_x = 0.0
                
            if abs(denom_y) > 1e-6:
                sub_y = 0.5 * (top - bottom) / denom_y
            else:
                sub_y = 0.0
                
            final_shift_x = coarse_shift_x + sub_x
            final_shift_y = coarse_shift_y + sub_y
        else:
            final_shift_x = float(coarse_shift_x)
            final_shift_y = float(coarse_shift_y)
    else:
        final_shift_x = float(coarse_shift_x)
        final_shift_y = float(coarse_shift_y)

    # Confidence metric: ratio of peak to mean/sigma of correlation
    # High peak indicates strong periodic similarity (good match)
    peak_val = corr_full[peak_y, peak_x]
    noise_floor = np.std(corr_full)
    if noise_floor < 1e-6:
        confidence = 0.0
    else:
        confidence = peak_val / noise_floor
        
    # Normalize confidence somewhat (heuristic)
    # Typical values: >5.0 is good, <2.0 is risky
    return final_shift_y, final_shift_x, confidence

def apply_shift_tensor(clip_tensor: torch.Tensor, shift_y: float, shift_x: float) -> torch.Tensor:
    """
    Apply a sub-pixel shift to a tensor using grid_sample.
    Args:
        clip_tensor: (1, C, H, W)
        shift_y, shift_x: Shift amounts in pixels
    Returns:
        shifted_tensor: (1, C, H, W)
    """
    if shift_y == 0.0 and shift_x == 0.0:
        return clip_tensor
        
    import torch.nn.functional as F
    
    b, c, h, w = clip_tensor.shape
    
    # Create grid
    y = torch.linspace(-1, 1, h, device=clip_tensor.device)
    x = torch.linspace(-1, 1, w, device=clip_tensor.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    # Grid is (H, W, 2) with values in [-1, 1]
    # Shift calculation:
    # New coordinate = Old coordinate - shift
    # In normalized space: shift_norm = shift / (size/2)
    
    shift_y_norm = shift_y / (h / 2.0)
    shift_x_norm = shift_x / (w / 2.0)
    
    # We want to sample from (x + shift), so the grid value at (y,x) 
    # should point to (x + shift_x_norm) in the source
    # Wait, grid_sample convention: output(y,x) samples input at grid(y,x)
    # If we want output(y,x) = input(y-shift, x-shift), then grid(y,x) must be (y-shift, x-shift)
    # Because we are shifting the IMAGE content by +shift, the pixel at new_pos comes from old_pos = new_pos - shift
    
    new_grid_x = grid_x - shift_x_norm
    new_grid_y = grid_y - shift_y_norm
    
    grid = torch.stack([new_grid_x, new_grid_y], dim=-1).unsqueeze(0) # (1, H, W, 2)
    
    # Clamp to border to avoid artifacts from out-of-bounds
    # grid_sample with padding_mode='border' handles this, but clamping grid explicitly is safer
    # grid = torch.clamp(grid, -1, 1) # Optional, grid_sample handles it
    
    shifted = F.grid_sample(
        clip_tensor, 
        grid, 
        mode='bicubic', 
        padding_mode='border', 
        align_corners=True
    )
    
    return shifted
