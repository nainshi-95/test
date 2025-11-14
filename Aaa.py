import torch
import numpy as np
import matplotlib.pyplot as plt

def get_dct_basis_flattened(block_size=8):
    """
    8x8 DCT Basis를 생성하여 (64, 8, 8) 형태로 반환합니다.
    Shape 설명: (Frequency_Index, Height, Width)
    """
    # 1. 1D DCT Matrix 생성 (Orthonormal)
    n = torch.arange(block_size).float()
    k = torch.arange(block_size).float()
    
    # cos((2n+1)kπ / 2N)
    dct_1d = torch.cos((2 * n.unsqueeze(1) + 1) * k.unsqueeze(0) * np.pi / (2 * block_size))
    
    # Normalization factor (alpha)
    alpha = torch.ones(block_size) * np.sqrt(2 / block_size)
    alpha[0] = np.sqrt(1 / block_size)
    
    # 1D Basis Matrix (Basis vectors are columns)
    # dct_matrix[k, n] 형태로 만듦 (k: frequency, n: time/space)
    dct_matrix = alpha.unsqueeze(1) * dct_1d.t() 
    
    # 2. 2D Basis 생성 (Outer Product)
    # (u, v) 주파수에 해당하는 2D 기저 이미지 생성
    basis_images = []
    
    for u in range(block_size):
        for v in range(block_size):
            # 1D basis vector for u and v
            basis_u = dct_matrix[u, :] # shape: (8,)
            basis_v = dct_matrix[v, :] # shape: (8,)
            
            # Outer product to make 8x8 basis image
            basis_img = torch.outer(basis_u, basis_v)
            basis_images.append(basis_img)
            
    # 3. Stack하여 (64, 8, 8) 텐서 생성
    # 순서는 (u=0,v=0), (u=0,v=1)... 순서로 flatten 됩니다.
    dct_basis_64_8_8 = torch.stack(basis_images)
    
    return dct_basis_64_8_8, dct_matrix

# --- 실행 및 검증 ---

# 1. 기저 생성
basis_tensor, T = get_dct_basis_flattened(8)
print(f"Basis Tensor Shape: {basis_tensor.shape}") # (64, 8, 8)

# 2. 8x8 입력 이미지 (Random Block)
original_block = torch.rand(8, 8) * 255
# 

# 3. Forward DCT (이미지 -> 계수)
# 방법: 입력 이미지와 각 기저 이미지의 내적(projection)을 구함
# 수식: Coeff[i] = sum(Image * Basis[i])
# view(-1, 64)는 이미지를 펼치는 것이 아니라, basis 연산을 위해 차원을 맞춤
# 여기서는 einsum을 사용하여 직관적으로 '이미지'와 '기저'를 곱해 '계수'를 뽑습니다.
coefficients = torch.einsum('hw, chw -> c', original_block, basis_tensor)

print(f"Coefficients Shape: {coefficients.shape}") # (64,) -> 64개의 DCT 계수

# 4. Inverse DCT (계수 -> 이미지 복원: 가중치 합)
# 사용자가 원한 방식: 계수(가중치) * Basis의 합
# 수식: Recon = sum(Coeff[i] * Basis[i])
reconstructed_block = torch.einsum('c, chw -> hw', coefficients, basis_tensor)

# 5. 검증 (원본과 복원본의 차이 확인)
mse = torch.nn.functional.mse_loss(original_block, reconstructed_block)
print(f"Reconstruction MSE Loss: {mse.item():.10f}") 
# 0에 매우 가까워야 함 (Floating point 오차 제외)

# 6. 시각화 (첫 5개 기저와 복원 결과)
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(original_block, cmap='gray')
plt.subplot(1, 3, 2)
plt.title("Reconstructed")
plt.imshow(reconstructed_block, cmap='gray')
plt.subplot(1, 3, 3)
plt.title("Basis #1 (Low Freq)")
plt.imshow(basis_tensor[1], cmap='gray') # u=0, v=1 basis
plt.show()
