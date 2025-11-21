import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os
from PIL import Image
import torchvision.transforms.functional as TF

# --- 1. VVC Luma Filter Definition ---
# 제공해주신 필터 계수 (16 phases, 8 taps)
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
        # 필터 텐서 생성 (Shape: 16 phases, 1 in_channel, 8 taps)
        coeffs = torch.tensor(vvc_filter_coeffs, dtype=torch.float32)
        
        # 정규화 (Sum이 64이므로 64로 나눔)
        coeffs = coeffs / 64.0
        
        # Conv2d에 사용하기 위해 weight shape 변환
        # Horizontal용: (Out=16, In=1, H=1, W=8) -> 각 픽셀마다 16개의 phase 생성
        self.weight_h = coeffs.view(16, 1, 1, 8)
        
        # Vertical용: (Out=16, In=1, H=8, W=1)
        self.weight_v = coeffs.view(16, 1, 8, 1)
        
        # Padding 설정 (8-tap 필터는 중심 기준 왼쪽 3, 오른쪽 4 패딩 필요)
        self.pad_h = (3, 4, 0, 0) # Left, Right, Top, Bottom
        self.pad_v = (0, 0, 3, 4) # Left, Right, Top, Bottom

    def forward(self, x):
        """
        x: (B, C, H, W)
        Separable Convolution을 수행하여 16배 업샘플링
        """
        b, c, h, w = x.shape
        
        # --- 1. Horizontal Upsampling (W -> 16W) ---
        # 채널별 연산을 위해 reshape: (B*C, 1, H, W)
        x_in = x.view(b * c, 1, h, w)
        
        # 가장자리 패딩 (Replication pad가 일반적)
        x_padded_h = F.pad(x_in, self.pad_h, mode='replicate')
        
        # Convolution 적용 -> 결과: (B*C, 16, H, W)
        # 각 픽셀 위치에서 16개의 phase 값(Sub-pixels)을 채널로 뽑아냄
        out_h = F.conv2d(x_padded_h, self.weight_h.to(x.device))
        
        # 채널 차원(16 phases)을 가로(Width) 차원으로 배치 (Pixel Shuffle 효과)
        # (B*C, 16, H, W) -> (B*C, H, W, 16) -> (B*C, H, W*16) -> (B*C, 1, H, W*16)
        out_h = out_h.permute(0, 2, 3, 1).reshape(b * c, 1, h, w * 16)
        
        # --- 2. Vertical Upsampling (H -> 16H) ---
        # 가장자리 패딩
        x_padded_v = F.pad(out_h, self.pad_v, mode='replicate')
        
        # Convolution 적용 -> 결과: (B*C, 16, H, 16W)
        out_v = F.conv2d(x_padded_v, self.weight_v.to(x.device))
        
        # 채널 차원(16 phases)을 세로(Height) 차원으로 배치
        # (B*C, 16, H, 16W) -> (B*C, H, 16W, 16) -> (B*C, H, 16, 16W) -> (B*C, H*16, 16W)
        # permute 순서 주의: (B*C, H, W_new, Phase)가 아니라 세로축 확장이므로 (B*C, H, Phase, W_new) 순서여야 함
        # 하지만 Conv 결과는 (N, C, H, W)이므로 -> (B*C, Phase, H, W_new)
        
        # View를 사용하여 (B*C, H, 16, W_new) 형태로 만들고 합침
        out_v = out_v.permute(0, 2, 1, 3).reshape(b * c, 1, h * 16, w * 16)
        
        # 원래 배치 크기로 복구
        return out_v.view(b, c, h * 16, w * 16)

def calculate_psnr(img1, img2):
    """
    img1, img2: Tensor or Numpy array (Values should be effectively 0-255 range)
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(255.0**2 / mse)

def run_test(kodak_path):
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 업샘플러 초기화
    upsampler = VVCUpsampler().to(device)
    
    # 이미지 파일 검색 (png, jpg 등)
    image_files = sorted(glob.glob(os.path.join(kodak_path, '*.*')))
    
    if not image_files:
        print("이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f"Found {len(image_files)} images.")
    print("-" * 60)
    print(f"{'Image Name':<30} | {'PSNR (dB)':<10}")
    print("-" * 60)

    avg_psnr = 0
    
    for img_path in image_files:
        # 이미지 로드 및 텐서 변환 (0~1 range, B,C,H,W)
        img = Image.open(img_path).convert('RGB')
        img_tensor = TF.to_tensor(img).unsqueeze(0).to(device) # (1, 3, H, W)
        
        # 1. 1/16 Bicubic Downsample
        # 원본 크기를 기억 (VVC 업샘플 후 비교를 위해)
        # H, W가 16의 배수가 아닐 경우를 대비해 Crop하거나, 
        # 다운샘플 시 align_corners=False 등을 고려해야 함. 
        # 여기서는 간단히 1/16로 줄이고 다시 16배 키움.
        
        _, _, H, W = img_tensor.shape
        # 크기를 16의 배수로 맞춤 (필요시) - 여기서는 생략하고 scale factor 사용
        
        down_h = H // 16
        down_w = W // 16
        
        # 1/16 크기로 다운샘플링
        downsampled = F.interpolate(img_tensor, size=(down_h, down_w), mode='bicubic', align_corners=False)
        
        # 2. VVC Filter Upsample (Custom Module)
        with torch.no_grad():
            upsampled_tensor = upsampler(downsampled)

        # 비교를 위해 원본 이미지도 16배수로 잘라줌 (다운샘플 시 버려진 픽셀 매칭)
        # 업샘플된 결과는 (down_h * 16, down_w * 16) 크기임
        target_h, target_w = upsampled_tensor.shape[2], upsampled_tensor.shape[3]
        original_cropped = img_tensor[:, :, :target_h, :target_w]
        
        # 3. PSNR 계산 (정수로 반올림 및 클리핑 후 계산)
        # 0~1 스케일을 0~255로 변환
        upsampled_int = torch.clamp(torch.round(upsampled_tensor * 255.0), 0, 255)
        original_int = torch.clamp(torch.round(original_cropped * 255.0), 0, 255)
        
        psnr = calculate_psnr(original_int, upsampled_int)
        
        img_name = os.path.basename(img_path)
        print(f"{img_name:<30} | {psnr.item():.4f}")
        avg_psnr += psnr.item()

    print("-" * 60)
    print(f"Average PSNR: {avg_psnr / len(image_files):.4f} dB")

# --- 실행 ---
# 아래 경로를 실제 Kodak 데이터셋 폴더 경로로 변경하세요.
# 예: './Kodak'
if __name__ == "__main__":
    # 예시 경로 (사용자 환경에 맞게 수정 필요)
    dataset_path = './kodak_dataset' 
    
    # 폴더가 없다면 경고
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_path}' 경로가 존재하지 않습니다. 코드를 수정하여 경로를 지정해주세요.")
    else:
        run_test(dataset_path)
