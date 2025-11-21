import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os
from PIL import Image
import math
import random

# --- Gaussian Blur Function (LPF) ---
def get_gaussian_kernel(kernel_size=5, sigma=1.0, channels=1):
    # 1D Gaussian kernel 생성
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Gaussian 수식 적용
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # 정규화 (Sum = 1)
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Conv2d 가중치 형태로 변환 (C, 1, K, K)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel

# --- Custom Dataset ---
class SuperResolutionDataset(Dataset):
    def __init__(self, hr_folder, patch_size=256, scale_factor=4):
        """
        hr_folder: HR 이미지가 있는 폴더 경로
        patch_size: 학습할 HR 이미지 크기 (기본 256)
        scale_factor: 다운샘플링 배율 (기본 4 -> 256/4 = 64)
        """
        self.image_files = sorted(glob.glob(os.path.join(hr_folder, '*.*')))
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        
        # Gaussian Kernel 미리 생성 (속도 최적화)
        # Scale 4 기준 Sigma=2.0, Kernel=9 정도가 적당 (Nyquist)
        self.sigma = scale_factor / 2.0
        self.kernel_size = int(math.ceil(self.sigma * 4)) | 1 
        self.gaussian_kernel = get_gaussian_kernel(self.kernel_size, self.sigma, channels=1)
        self.padding = self.kernel_size // 2

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. 이미지 로드 (Y channel만 사용한다고 가정, 필요시 RGB로 변경 가능)
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('YCbCr')
        y, _, _ = img.split() # Y 채널만 추출
        
        # Tensor 변환 (0~1)
        img_tensor = transforms.ToTensor()(y) # (1, H, W)

        # 2. Random Crop (256x256)
        # 원본 이미지가 patch_size보다 작으면 에러나므로 패딩하거나 resize해야 함
        # 여기서는 원본이 충분히 크다고 가정
        C, H, W = img_tensor.shape
        
        if H < self.patch_size or W < self.patch_size:
            # 이미지가 작으면 강제로 256으로 늘림 (예외처리)
             img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(self.patch_size, self.patch_size), mode='bicubic').squeeze(0)
             H, W = self.patch_size, self.patch_size

        # 랜덤 좌표 계산 (Scale의 배수로 맞춰야 정확한 Decimation 가능)
        # LR과 위상을 맞추기 위해 crop 좌표도 scale의 배수가 되도록 설정
        h_start = random.randrange(0, H - self.patch_size + 1, self.scale_factor)
        w_start = random.randrange(0, W - self.patch_size + 1, self.scale_factor)
        
        hr_patch = img_tensor[:, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]

        # 3. Downsampling (LPF + Decimation)
        # Conv2d를 위해 배치 차원 추가 (1, 1, H, W)
        hr_input = hr_patch.unsqueeze(0)
        
        # Reflection padding으로 경계 부작용 최소화
        hr_padded = F.pad(hr_input, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        
        # Gaussian Blur
        # functional interface 사용
        blurred = F.conv2d(hr_padded, self.gaussian_kernel, groups=1)
        
        # Pixel Sampling (Decimation)
        # ::scale_factor 슬라이싱 사용
        lr_patch = blurred[:, :, ::self.scale_factor, ::self.scale_factor]
        
        # 배치 차원 제거 (C, H, W)
        lr_patch = lr_patch.squeeze(0)

        return lr_patch, hr_patch

# --- 실행 예시 ---
if __name__ == "__main__":
    # 설정
    HR_FOLDER = './kodak_dataset' # 실제 경로로 변경
    BATCH_SIZE = 16
    
    # 데이터셋 및 로더 초기화
    dataset = SuperResolutionDataset(hr_folder=HR_FOLDER, patch_size=256, scale_factor=4)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 학습 루프 예시
    print(f"Total Images: {len(dataset)}")
    
    # 배치 하나 뽑아서 확인
    for i, (lr, hr) in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"LR Shape: {lr.shape}") # (Batch, 1, 64, 64) 예상
        print(f"HR Shape: {hr.shape}") # (Batch, 1, 256, 256) 예상
        
        # 실제 학습 코드...
        # optimizer.zero_grad()
        # output = model(lr)
        # loss = criterion(output, hr)
        # ...
        
        break # 테스트용 break

