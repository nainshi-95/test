import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def get_jpeg_quantization_matrix(quality):
    """
    JPEG 표준 Luminance(휘도) 양자화 테이블
    """
    Q_luminance = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    if quality <= 0: quality = 1
    if quality > 100: quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    Q_scaled = np.floor((Q_luminance * scale + 50) / 100)
    Q_scaled[Q_scaled < 1] = 1
    Q_scaled[Q_scaled > 255] = 255

    return Q_scaled

def process_8x8_block(block, Q_matrix):
    """
    단일 8x8 블록 처리 (DCT -> Quant -> Dequant -> IDCT)
    """
    # 1. Level Shift (-128)
    block_shifted = block.astype(np.float32) - 128.0
    
    # 2. DCT
    dct_coeff = cv2.dct(block_shifted)
    
    # 3. Quantization
    quantized_coeff = np.round(dct_coeff / Q_matrix)
    
    # 4. Dequantization
    recon_dct = quantized_coeff * Q_matrix
    
    # 5. IDCT
    recon_block_shifted = cv2.idct(recon_dct)
    
    # 6. Level Shift Back (+128)
    recon_block = recon_block_shifted + 128.0
    
    # 7. Clipping
    return np.clip(recon_block, 0, 255)

def jpeg_simulate_y_domain(image_path, quality=50):
    # 1. 이미지 로드
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("이미지를 찾을 수 없어 임의의 이미지를 생성합니다.")
        img_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.randn(img_bgr, (128, 128, 128), (50, 50, 50)) # 노이즈 이미지 생성

    h, w = img_bgr.shape[:2]

    # 2. BGR -> YCrCb 변환
    # OpenCV에서 YCrCb 순서: 채널 0=Y(Luma), 1=Cr, 2=Cb
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    
    # 채널 분리
    y, cr, cb = cv2.split(img_ycrcb)

    # 3. 패딩 (8의 배수로 맞춤) - Y, Cr, Cb 모두 동일하게 적용해야 합쳐짐
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    # Y 채널만 압축 과정을 거칠 것이므로, Y에 패딩 적용
    y_padded = cv2.copyMakeBorder(y, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    
    # Cr, Cb는 압축 안 하지만 크기는 맞춰야 나중에 합칠 수 있음
    cr_padded = cv2.copyMakeBorder(cr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    cb_padded = cv2.copyMakeBorder(cb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    padded_h, padded_w = y_padded.shape

    # 결과 담을 배열
    y_reconstructed = np.zeros_like(y_padded, dtype=np.float32)
    
    # 양자화 행렬 준비
    Q_matrix = get_jpeg_quantization_matrix(quality)

    # 4. Y 채널 블록 루프 (압축 수행)
    for i in range(0, padded_h, 8):
        for j in range(0, padded_w, 8):
            block = y_padded[i:i+8, j:j+8]
            
            # 블록 처리 (DCT 과정)
            recon_block = process_8x8_block(block, Q_matrix)
            
            y_reconstructed[i:i+8, j:j+8] = recon_block

    # uint8 변환
    y_reconstructed = y_reconstructed.astype(np.uint8)

    # 5. 채널 병합 (압축된 Y + 원본 Cr + 원본 Cb)
    # 실제 JPEG는 Cr, Cb도 서브샘플링하고 양자화하지만, 여기선 Y만 흉내냅니다.
    merged_ycrcb = cv2.merge([y_reconstructed, cr_padded, cb_padded])

    # 6. 패딩 제거 (원본 크기로 자르기)
    merged_ycrcb = merged_ycrcb[:h, :w]

    # 7. YCrCb -> BGR 변환 (최종 결과)
    final_bgr = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)

    return img_bgr, final_bgr, y, y_reconstructed


def main():
    # --- 실행 코드 ---
    # 'test_image.jpg'가 있다면 사용하고, 없으면 위에서 생성된 노이즈 이미지를 사용
    img_path = "A:/python_code/DB/Kodak/1.png"
    original, compressed, y, y_reconstructed = jpeg_simulate_y_domain(img_path, quality=10)

    Image.fromarray(y).save("original_Y.jpg")
    Image.fromarray(y_reconstructed).save("recon_Y.jpg")


if __name__ == "__main__":
    main()
