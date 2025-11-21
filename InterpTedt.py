import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
import argparse
from PIL import Image
import torchvision.transforms.functional as TF

# --- VVC Luma Filter Definition (16 Phases, 8 Taps) ---
vvc_filter_coeffs = [
  [  0, 0,   0, 64,  0,   0,  0,  0 ], # Phase 0 (Integer position)
  [  0, 1,  -3, 63,  4,  -2,  1,  0 ], # Phase 1
  [ -1, 2,  -5, 62,  8,  -3,  1,  0 ], # Phase 2
  [ -1, 3,  -8, 60, 13,  -4,  1,  0 ], # Phase 3
  [ -1, 4, -10, 58, 17,  -5,  1,  0 ], # Phase 4 (1/4 position)
  [ -1, 4, -11, 52, 26,  -8,  3, -1 ], # Phase 5
  [ -1, 3,  -9, 47, 31, -10,  4, -1 ], # Phase 6
  [ -1, 4, -11, 45, 34, -10,  4, -1 ], # Phase 7
  [ -1, 4, -11, 40, 40, -11,  4, -1 ], # Phase 8 (1/2 position)
  [ -1, 4, -10, 34, 45, -11,  4, -1 ], # Phase 9
  [ -1, 4, -10, 31, 47,  -9,  3, -1 ], # Phase 10
  [ -1, 3,  -8, 26, 52, -11,  4, -1 ], # Phase 11
  [  0, 1,  -5, 17, 58, -10,  4, -1 ], # Phase 12 (3/4 position)
  [  0, 1,  -4, 13, 60,  -8,  3, -1 ], # Phase 13
  [  0, 1,  -3,  8, 62,  -5,  2, -1 ], # Phase 14
  [  0, 1,  -2,  4, 63,  -3,  1,  0 ]  # Phase 15
]

class VVCUpsampler(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 스케일 팩터 검증
        if scale_factor not in [2, 4, 8, 16]:
            raise ValueError("Scale factor must be 2, 4, 8, or 16.")

        # 필요한 Phase 선택 로직 (Polyphase Subsampling)
        # 예: scale=2 -> step=8 -> indices [0, 8] 사용
        # 예: scale=4 -> step=4 -> indices [0, 4, 8, 12] 사용
        step = 16 // scale_factor
        indices = list(range(0, 16, step))
        
        full_coeffs = torch.tensor(vvc_filter_coeffs, dtype=torch.float32)
        selected_coeffs = full_coeffs[indices] # (scale_factor, 8)
        
        # 정규화
        coeffs = selected_coeffs / 64.0
        
        # Conv2d Weight 설정
        # Horizontal: (Out=scale, In=1, H=1, W=8)
        self.weight_h = coeffs.view(scale_factor, 1, 1, 8)
        # Vertical: (Out=scale, In=1, H=8, W=1)
        self.weight_v = coeffs.view(scale_factor, 1, 8, 1)
        
        # Padding (8-tap 필터 고정)
        self.pad_h = (3, 4, 0, 0)
        self.pad_v = (0, 0, 3, 4)

    def forward(self, x):
        b, c, h, w = x.shape
        scale = self.scale_factor
        
        # --- Horizontal Upsampling ---
        x_in = x.view(b * c, 1, h, w)
        x_padded_h = F.pad(x_in, self.pad_h, mode='replicate')
        
        # 결과: (B*C, scale, H, W)
        out_h = F.conv2d(x_padded_h, self.weight_h.to(x.device))
        
        # Pixel Shuffle (Width 방향 확장)
        # (B*C, scale, H, W) -> (B*C, 1, H, W * scale)
        out_h = out_h.permute(0, 2, 3, 1).reshape(b * c, 1, h, w * scale)
        
        # --- Vertical Upsampling ---
        x_padded_v = F.pad(out_h, self.pad_v, mode='replicate')
        
        # 결과: (B*C, scale, H, W*scale)
        out_v = F.conv2d(x_padded_v, self.weight_v.to(x.device))
        
        # Pixel Shuffle (Height 방향 확장)
        # (B*C, scale, H, W*scale) -> (B*C, 1, H * scale, W * scale)
        out_v = out_v.permute(0, 2, 1, 3).reshape(b * c, 1, h * scale, w * scale)
        
        return out_v.view(b, c, h * scale, w * scale)

def calculate_psnr_y_only(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 10 * torch.log10(255.0**2 / mse)

def save_image_tensor(tensor, path):
    """
    tensor: (1, H, W) float tensor (0~1)
    path: save path
    """
    # CPU로 이동 후 PIL 변환 및 저장
    tf_img = TF.to_pil_image(tensor.cpu())
    tf_img.save(path)

def run_test(kodak_path, save_dir, scale_factor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Testing Scale Factor: x{scale_factor}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created output directory: {save_dir}")

    # 선택된 스케일에 맞는 업샘플러 초기화
    upsampler = VVCUpsampler(scale_factor).to(device)
    
    image_files = sorted(glob.glob(os.path.join(kodak_path, '*.*')))
    if not image_files:
        print("이미지를 찾을 수 없습니다.")
        return

    print("-" * 80)
    print(f"{'Image Name':<25} | {'Y-PSNR (dB)':<10} | Saved (Orig, Down, Up)")
    print("-" * 80)

    avg_psnr = 0
    
    for img_path in image_files:
        # 1. 로드 및 Y 채널 추출
        img = Image.open(img_path).convert('YCbCr')
        y, _, _ = img.split()
        img_tensor = TF.to_tensor(y).unsqueeze(0).to(device) # (1, 1, H, W)
        
        # 크기 보정 (스케일의 배수가 되도록 자름, 정확한 PSNR 측정을 위해)
        _, _, H, W = img_tensor.shape
        H_crop = (H // scale_factor) * scale_factor
        W_crop = (W // scale_factor) * scale_factor
        img_tensor = img_tensor[:, :, :H_crop, :W_crop]
        
        # 2. 다운샘플링 (Bicubic + Antialias)
        down_h = H_crop // scale_factor
        down_w = W_crop // scale_factor
        
        downsampled = F.interpolate(
            img_tensor, 
            size=(down_h, down_w), 
            mode='bicubic', 
            align_corners=False, 
            antialias=True
        )
        
        # 3. VVC 업샘플링
        with torch.no_grad():
            upsampled_tensor = upsampler(downsampled)

        # 4. PSNR 계산 (정수 반올림)
        # img_tensor는 이미 crop된 상태이므로 바로 비교 가능
        upsampled_int = torch.clamp(torch.round(upsampled_tensor * 255.0), 0, 255)
        original_int = torch.clamp(torch.round(img_tensor * 255.0), 0, 255)
        
        psnr = calculate_psnr_y_only(original_int, upsampled_int)
        
        # 5. 이미지 저장 (Original Y, Downsampled Y, Upsampled Y)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 저장 경로 설정
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
    # Argument Parser 설정
    parser = argparse.ArgumentParser(description='VVC Upsampling Test')
    parser.add_argument('--scale', type=int, default=16, choices=[2, 4, 8, 16], 
                        help='Upsampling scale factor (2, 4, 8, 16). Default is 16.')
    parser.add_argument('--input_dir', type=str, default='./kodak_dataset', 
                        help='Input dataset directory')
    parser.add_argument('--output_dir', type=str, default='./result_images', 
                        help='Output directory for images')

    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
    else:
        run_test(args.input_dir, args.output_dir, args.scale)
