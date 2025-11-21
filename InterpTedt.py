import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os
from PIL import Image
import torchvision.transforms.functional as TF

# --- 1. VVC Luma Filter Definition (변경 없음) ---
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
  [  0, 1,  -5, 17, 58, -10,  4, -1 },
  [  0, 1,  -4, 13, 60,  -8,  3, -1 },
  [  0, 1,  -3,  8, 62,  -5,  2, -1 },
  [  0, 1,  -2,  4, 63,  -3,  1,  0 ]
]

class VVCUpsampler(nn.Module):
    def __init__(self):
        super().__init__()
        coeffs = torch.tensor(vvc_filter_coeffs, dtype=torch.float32)
        coeffs = coeffs / 64.0
        
        # Horizontal: (Out=16, In=1, H=1, W=8)
        self.weight_h = coeffs.view(16, 1, 1, 8)
        # Vertical: (Out=16, In=1, H=8, W=1)
        self.weight_v = coeffs.view(16, 1, 8, 1)
        
        self.pad_h = (3, 4, 0, 0)
        self.pad_v = (0, 0, 3, 4)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # --- Horizontal ---
        x_in = x.view(b * c, 1, h, w)
        x_padded_h = F.pad(x_in, self.pad_h, mode='replicate')
        out_h = F.conv2d(x_padded_h, self.weight_h.to(x.device))
        out_h = out_h.permute(0, 2, 3, 1).reshape(b * c, 1, h, w * 16)
        
        # --- Vertical ---
        x_padded_v = F.pad(out_h, self.pad_v, mode='replicate')
        out_v = F.conv2d(x_padded_v, self.weight_v.to(x.device))
        out_v = out_v.permute(0, 2, 1, 3).reshape(b * c, 1, h * 16, w * 16)
        
        return out_v.view(b, c, h * 16, w * 16)

def calculate_psnr_y_only(img1, img2):
    # Y 채널만 있는 상태 (0~255)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(255.0**2 / mse)

def run_test(kodak_path, save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 저장 폴더 생성
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created output directory: {save_dir}")

    upsampler = VVCUpsampler().to(device)
    image_files = sorted(glob.glob(os.path.join(kodak_path, '*.*')))
    
    if not image_files:
        print("이미지를 찾을 수 없습니다.")
        return

    print("-" * 70)
    print(f"{'Image Name':<30} | {'Y-PSNR (dB)':<10} | Saved")
    print("-" * 70)

    avg_psnr = 0
    
    for img_path in image_files:
        # 1. YCbCr 변환 후 Y 채널만 추출
        img = Image.open(img_path).convert('YCbCr')
        y, _, _ = img.split() # Cb, Cr은 여기서 버림
        
        # (1, 1, H, W) 텐서로 변환
        img_tensor = TF.to_tensor(y).unsqueeze(0).to(device)
        
        # 2. 1/16 Bicubic Downsample
        _, _, H, W = img_tensor.shape
        down_h = H // 16
        down_w = W // 16
        
        # 다운샘플링
        downsampled = F.interpolate(img_tensor, size=(down_h, down_w), mode='bicubic', align_corners=False)
        
        # 3. VVC Filter Upsample (Y Only)
        with torch.no_grad():
            upsampled_tensor = upsampler(downsampled)

        # 원본 크기 맞춰 자르기 (비교용)
        target_h, target_w = upsampled_tensor.shape[2], upsampled_tensor.shape[3]
        original_cropped = img_tensor[:, :, :target_h, :target_w]
        
        # 4. PSNR 계산 (정수 반올림)
        upsampled_int = torch.clamp(torch.round(upsampled_tensor * 255.0), 0, 255)
        original_int = torch.clamp(torch.round(original_cropped * 255.0), 0, 255)
        
        psnr = calculate_psnr_y_only(original_int, upsampled_int)
        
        # 5. 결과 이미지 저장 (Grayscale)
        # upsampled_int는 float 타입이므로 byte로 변환하여 이미지 생성
        save_name = os.path.splitext(os.path.basename(img_path))[0] + "_y_upsampled.png"
        save_full_path = os.path.join(save_dir, save_name)
        
        # Tensor -> PIL Image (L mode)
        # squeeze(0)으로 batch 제거 -> (1, H, W) -> to_pil_image가 알아서 L모드 처리
        out_img = TF.to_pil_image(upsampled_tensor.squeeze(0).cpu())
        out_img.save(save_full_path)
        
        print(f"{os.path.basename(img_path):<30} | {psnr.item():.4f}      | Yes")
        avg_psnr += psnr.item()

    print("-" * 70)
    print(f"Average Y-PSNR: {avg_psnr / len(image_files):.4f} dB")
    print(f"All results saved to: {save_dir}")

if __name__ == "__main__":
    # 입력 데이터셋 경로
    dataset_path = './kodak_dataset'
    
    # 결과 저장 경로
    output_path = './result_y_images'
    
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_path}' 경로가 존재하지 않습니다.")
    else:
        run_test(dataset_path, output_path)
