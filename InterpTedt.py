import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
import argparse
import math
from PIL import Image
import torchvision.transforms.functional as TF

# --- VVC Luma Filter Definition ---
vvc_filter_coeffs = [
  [  0, 0,   0, 64,  0,   0,  0,  0 ],
  [  0, 1,  -3, 63,  4,  -2,  1,  0 ],
  [ -1, 2,  -5, 62,  8,  -3,  1,  0 ],
  [ -1, 3,  -8, 60, 13,  -4,  1,  0 ],
  [ -1, 4, -10, 58, 17,  -5,  1,  0 ],
  [ -1, 4, -11, 52, 26,  -8,  3, -1 ],
  [ -1, 3,  -9, 47, 31, -10,  4, -1 ],
  [ -1, 4, -11, 45, 34, -10,  4, -1 ],
  [ -1, 4, -11, 40, 40, -11,  4, -1 ],
  [ -1, 4, -10, 34, 45, -11,  4, -1 ],
  [ -1, 4, -10, 31, 47,  -9,  3, -1 ],
  [ -1, 3,  -8, 26, 52, -11,  4, -1 ],
  [  0, 1,  -5, 17, 58, -10,  4, -1 ],
  [  0, 1,  -4, 13, 60,  -8,  3, -1 ],
  [  0, 1,  -3,  8, 62,  -5,  2, -1 ],
  [  0, 1,  -2,  4, 63,  -3,  1,  0 ]
]

class VVCUpsampler(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        
        if scale_factor not in [2, 4, 8, 16]:
            raise ValueError("Scale factor must be 2, 4, 8, or 16.")

        step = 16 // scale_factor
        indices = list(range(0, 16, step))
        
        full_coeffs = torch.tensor(vvc_filter_coeffs, dtype=torch.float32)
        selected_coeffs = full_coeffs[indices]
        
        coeffs = selected_coeffs / 64.0
        
        self.weight_h = coeffs.view(scale_factor, 1, 1, 8)
        self.weight_v = coeffs.view(scale_factor, 1, 8, 1)
        
        # 정수 위치 샘플링을 했으므로, VVC 필터의 이론적 패딩값인 (3, 4)가 정확히 맞습니다.
        self.pad_h = (3, 4, 0, 0)
        self.pad_v = (0, 0, 3, 4)

    def forward(self, x):
        b, c, h, w = x.shape
        scale = self.scale_factor
        
        x_in = x.view(b * c, 1, h, w)
        x_padded_h = F.pad(x_in, self.pad_h, mode='replicate')
        out_h = F.conv2d(x_padded_h, self.weight_h.to(x.device))
        out_h = out_h.permute(0, 2, 3, 1).reshape(b * c, 1, h, w * scale)
        
        x_padded_v = F.pad(out_h, self.pad_v, mode='replicate')
        out_v = F.conv2d(x_padded_v, self.weight_v.to(x.device))
        out_v = out_v.permute(0, 2, 1, 3).reshape(b * c, 1, h * scale, w * scale)
        
        return out_v.view(b, c, h * scale, w * scale)

# --- Gaussian Blur Function ---
def apply_gaussian_blur(img, kernel_size, sigma):
    """
    img: (B, C, H, W)
    kernel_size: int (odd number)
    sigma: float
    """
    # Create a 1D Gaussian kernel
    k = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) // 2
    gaussian_1d = torch.exp(-0.5 * (k / sigma) ** 2)
    gaussian_1d = gaussian_1d / gaussian_1d.sum() # Normalize
    
    # Create 2D kernel (separable)
    # (Out, In, H, W) -> (1, 1, K, 1) and (1, 1, 1, K)
    kernel_x = gaussian_1d.view(1, 1, 1, kernel_size).to(img.device)
    kernel_y = gaussian_1d.view(1, 1, kernel_size, 1).to(img.device)
    
    # Padding size
    pad = kernel_size // 2
    
    # Apply separable convolution
    # Replicate padding to reduce boundary artifacts
    img_padded_x = F.pad(img, (pad, pad, 0, 0), mode='replicate')
    img_blur_x = F.conv2d(img_padded_x, kernel_x)
    
    img_padded_y = F.pad(img_blur_x, (0, 0, pad, pad), mode='replicate')
    img_blur = F.conv2d(img_padded_y, kernel_y)
    
    return img_blur

def calculate_psnr_y_only(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 10 * torch.log10(255.0**2 / mse)

def save_image_tensor(tensor, path):
    tf_img = TF.to_pil_image(tensor.cpu())
    tf_img.save(path)

def run_test(kodak_path, save_dir, scale_factor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Testing Scale Factor: x{scale_factor}")
    print("Method: Gaussian LPF -> Integer Sampling (Decimation)")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    upsampler = VVCUpsampler(scale_factor).to(device)
    image_files = sorted(glob.glob(os.path.join(kodak_path, '*.*')))
    
    if not image_files:
        print("이미지를 찾을 수 없습니다.")
        return

    print("-" * 80)
    print(f"{'Image Name':<25} | {'Y-PSNR (dB)':<10} | Saved")
    print("-" * 80)

    avg_psnr = 0
    
    for img_path in image_files:
        img = Image.open(img_path).convert('YCbCr')
        y, _, _ = img.split()
        img_tensor = TF.to_tensor(y).unsqueeze(0).to(device)
        
        _, _, H, W = img_tensor.shape
        # scale의 배수로 Crop
        H_crop = (H // scale_factor) * scale_factor
        W_crop = (W // scale_factor) * scale_factor
        img_tensor = img_tensor[:, :, :H_crop, :W_crop]
        
        # --- 1. Low Pass Filter (Gaussian) ---
        # Sigma는 보통 scale factor에 비례하게 설정 (Nyquist 이론)
        # sigma = scale / 2 정도가 적당함.
        sigma = scale_factor / 2.0
        # Kernel size는 sigma의 약 4~6배 (홀수)
        kernel_size = int(math.ceil(sigma * 4)) | 1 
        
        blurred_img = apply_gaussian_blur(img_tensor, kernel_size, sigma)
        
        # --- 2. Integer Grid Sampling (Decimation) ---
        # 정확히 0, scale, 2*scale 위치의 픽셀만 가져옴
        downsampled = blurred_img[:, :, ::scale_factor, ::scale_factor]
        
        # --- 3. VVC Upsampling ---
        with torch.no_grad():
            upsampled_tensor = upsampler(downsampled)

        # PSNR Calculation
        upsampled_int = torch.clamp(torch.round(upsampled_tensor * 255.0), 0, 255)
        original_int = torch.clamp(torch.round(img_tensor * 255.0), 0, 255)
        
        psnr = calculate_psnr_y_only(original_int, upsampled_int)
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        path_orig = os.path.join(save_dir, f"{base_name}_orig_y.png")
        path_down = os.path.join(save_dir, f"{base_name}_down_x{scale_factor}.png")
        path_up   = os.path.join(save_dir, f"{base_name}_up_x{scale_factor}.png")
        
        save_image_tensor(img_tensor.squeeze(0), path_orig)
        save_image_tensor(downsampled.squeeze(0), path_down)
        save_image_tensor(upsampled_tensor.squeeze(0), path_up)
        
        print(f"{base_name:<25} | {psnr.item():.4f}      | Yes")
        avg_psnr += psnr.item()

    print("-" * 80)
    print(f"Average Y-PSNR (x{scale_factor}): {avg_psnr / len(image_files):.4f} dB")
    print(f"Result images saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=16, choices=[2, 4, 8, 16])
    parser.add_argument('--input_dir', type=str, default='./kodak_dataset')
    parser.add_argument('--output_dir', type=str, default='./result_images')
    args = parser.parse_args()
    
    if os.path.exists(args.input_dir):
        run_test(args.input_dir, args.output_dir, args.scale)
    else:
        print("Input directory not found.")
