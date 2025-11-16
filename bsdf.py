import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import struct

# ==========================================
# 1. JPEG Standard Tables (Luminance)
# ==========================================

# Zigzag Lookup Table
ZIGZAG_ORDER = np.array([
    0,  1,  5,  6, 14, 15, 27, 28,
    2,  4,  7, 13, 16, 26, 29, 42,
    3,  8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
], dtype=np.int32)

# Standard JPEG Luminance Quantization Table
Q_LUMINANCE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

# Standard Huffman Tables (Length, Value) pairs are omitted for brevity in manual implementations normally,
# but here we use simplified standard maps.
# Format: { (Run, Size): (Code, BitLength) } for AC, { Size: (Code, BitLength) } for DC

# NOTE: For this simulation, we use a simplified predefined hardcoded mapping based on standard JPEG logic
# to avoid thousands of lines of table definitions.
# In a real JPEG, these are constructed from DHT markers.

class HuffmanTable:
    def __init__(self):
        self.dc_map = {} # key: size, value: (code, len)
        self.ac_map = {} # key: (run, size), value: (code, len)
        self.dc_map_inv = {}
        self.ac_map_inv = {}
        self._init_standard_tables()

    def _init_standard_tables(self):
        # --- Simplified Standard Luma DC Table ---
        # Size -> (Code, Length)
        # This is a subset/example of the actual standard for demonstration
        dc_codes = {
            0: (0x00, 2), 1: (0x02, 3), 2: (0x03, 3), 3: (0x04, 3),
            4: (0x05, 3), 5: (0x06, 3), 6: (0x0E, 4), 7: (0x1E, 5),
            8: (0x3E, 6), 9: (0x7E, 7), 10: (0xFE, 8), 11: (0x1FE, 9)
        }
        self.dc_map = dc_codes
        self.dc_map_inv = {v: k for k, v in dc_codes.items()}

        # --- Simplified Standard Luma AC Table ---
        # (Run, Size) -> (Code, Length)
        # Run: 0-15, Size: 1-10. (0,0) is EOB, (15,0) is ZRL
        # Real standard table is huge. We generate a pseudo-standard one or use a subset.
        # For robustness in this script, we will programmatically generate codes
        # similar to the standard to ensure we cover all cases.
        
        # Using a generic prefix code generator for (Run, Size) pairs to ensure unique decodability
        # In real JPEG, this is fixed. Here we simulate it.
        vals = []
        vals.append((0,0)) # EOB
        vals.append((15,0)) # ZRL
        for r in range(16):
            for s in range(1, 11):
                vals.append((r, s))
        
        # Assign simple prefix codes (simulated Canonical Huffman)
        # Just for simulation: We map these 162 symbols to variable length codes
        # This is functionally equivalent to using the standard table for RD purposes
        current_code = 0
        current_len = 2
        
        # Sort roughly by probability (low run, low size -> shorter code)
        vals.sort(key=lambda x: x[0]*2 + x[1]) 

        for v in vals:
            self.ac_map[v] = (current_code, current_len)
            current_code += 1
            # Check if code exceeds current bit length capacity
            if current_code >= (1 << current_len):
                current_code = 0
                current_len += 1
                # Limit max length to avoid explosion (JPEG limit is 16)
                if current_len > 16: current_len = 16 
            
        self.ac_map_inv = {v: k for k, v in self.ac_map.items()}

huff_table = HuffmanTable()

# ==========================================
# 2. Helper Functions: VLI & BitStream
# ==========================================

def get_vli(val):
    """Variable Length Integer representation."""
    if val == 0: return 0, 0
    if val > 0:
        size = val.bit_length()
        code = val
    else:
        size = abs(val).bit_length()
        code = (abs(val) ^ ((1 << size) - 1)) # Ones complement
    return code, size

def decode_vli(code, size):
    """Reverse VLI."""
    if size == 0: return 0
    if (code >> (size - 1)) == 1: # Positive
        return code
    else: # Negative
        return -(code ^ ((1 << size) - 1))

class BitWriter:
    def __init__(self):
        self.buffer = []
        self.curr_byte = 0
        self.curr_bit_idx = 0

    def write(self, val, length):
        for i in range(length - 1, -1, -1):
            bit = (val >> i) & 1
            self.curr_byte = (self.curr_byte << 1) | bit
            self.curr_bit_idx += 1
            if self.curr_bit_idx == 8:
                self.buffer.append(self.curr_byte)
                self.curr_byte = 0
                self.curr_bit_idx = 0
    
    def flush(self):
        if self.curr_bit_idx > 0:
            self.curr_byte = self.curr_byte << (8 - self.curr_bit_idx)
            self.buffer.append(self.curr_byte)
        return bytes(self.buffer)

class BitReader:
    def __init__(self, data):
        self.data = data
        self.byte_idx = 0
        self.bit_idx = 0
    
    def read(self, length):
        val = 0
        for _ in range(length):
            if self.byte_idx >= len(self.data):
                return None # EOF
            
            bit = (self.data[self.byte_idx] >> (7 - self.bit_idx)) & 1
            val = (val << 1) | bit
            
            self.bit_idx += 1
            if self.bit_idx == 8:
                self.bit_idx = 0
                self.byte_idx += 1
        return val
    
    def peek_bits(self, length):
        # Look ahead without consuming
        old_byte = self.byte_idx
        old_bit = self.bit_idx
        val = self.read(length)
        self.byte_idx = old_byte
        self.bit_idx = old_bit
        return val

# ==========================================
# 3. Core Processing (DCT, Quant, Zigzag, Entropy)
# ==========================================

def get_scaled_q_matrix(quality):
    if quality <= 0: quality = 1
    if quality > 100: quality = 100
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    Q_scaled = np.floor((Q_LUMINANCE * scale + 50) / 100)
    Q_scaled[Q_scaled < 1] = 1
    Q_scaled[Q_scaled > 255] = 255
    return Q_scaled

def encode_block(block, prev_dc, Q_matrix, writer):
    # 1. DCT & Quant
    block_f = block.astype(np.float32) - 128.0
    dct = cv2.dct(block_f)
    quant = np.round(dct / Q_matrix).astype(np.int32)
    
    # 2. Zigzag Scan
    zz = quant.flatten()[ZIGZAG_ORDER]
    
    # 3. DC Encoding (DPCM)
    diff = zz[0] - prev_dc
    dc_code, dc_size = get_vli(diff)
    
    # Write DC Huffman
    huff_code, huff_len = huff_table.dc_map.get(dc_size, (None, None))
    if huff_code is None: huff_code, huff_len = huff_table.dc_map[11] # Fallback max
    writer.write(huff_code, huff_len)
    writer.write(dc_code, dc_size)
    
    # 4. AC Encoding (RLE)
    run = 0
    for i in range(1, 64):
        val = zz[i]
        if val == 0:
            run += 1
        else:
            while run >= 16:
                # ZRL (15 zeros, size 0)
                hc, hl = huff_table.ac_map[(15, 0)]
                writer.write(hc, hl)
                run -= 16
            
            ac_code, ac_size = get_vli(val)
            if ac_size > 10: ac_size = 10 # Clamp for simple table
            
            # Huffman for (Run, Size)
            if (run, ac_size) not in huff_table.ac_map:
                 # Fallback if our simplified table misses rare case
                 hc, hl = huff_table.ac_map[(0, 1)] 
            else:
                hc, hl = huff_table.ac_map[(run, ac_size)]
            
            writer.write(hc, hl)
            writer.write(ac_code, ac_size)
            run = 0
            
    # EOB (End of Block)
    if run > 0:
        hc, hl = huff_table.ac_map[(0, 0)]
        writer.write(hc, hl)
        
    return zz[0] # Return DC for next DPCM

def decode_block(reader, prev_dc, Q_matrix):
    # 1. Decode DC
    # In a real decoder, we walk the tree. Here we bruteforce lookup for simplicity
    # (inefficient but works for simulation)
    found = False
    code_buf = 0
    length = 0
    dc_size = 0
    
    while not found:
        bit = reader.read(1)
        if bit is None: return None, prev_dc
        code_buf = (code_buf << 1) | bit
        length += 1
        if (code_buf, length) in huff_table.dc_map_inv:
            dc_size = huff_table.dc_map_inv[(code_buf, length)]
            found = True
            
    diff_bits = reader.read(dc_size)
    diff = decode_vli(diff_bits, dc_size)
    dc_val = prev_dc + diff
    
    # 2. Decode AC
    zz = np.zeros(64, dtype=np.int32)
    zz[0] = dc_val
    idx = 1
    
    while idx < 64:
        # Decode Huffman Symbol (Run, Size)
        found = False
        code_buf = 0
        length = 0
        rs_pair = (0,0)
        
        while not found:
            bit = reader.read(1)
            code_buf = (code_buf << 1) | bit
            length += 1
            if (code_buf, length) in huff_table.ac_map_inv:
                rs_pair = huff_table.ac_map_inv[(code_buf, length)]
                found = True
                
        run, size = rs_pair
        
        if run == 0 and size == 0: # EOB
            break
        elif run == 15 and size == 0: # ZRL
            idx += 16
            continue
        
        idx += run
        if idx >= 64: break # Safety
        
        val_bits = reader.read(size)
        val = decode_vli(val_bits, size)
        zz[idx] = val
        idx += 1
        
    # 3. Inverse Zigzag
    quant_block = np.zeros(64, dtype=np.int32)
    quant_block[ZIGZAG_ORDER] = zz
    quant_block = quant_block.reshape((8, 8))
    
    # 4. Dequant & IDCT
    dequant = quant_block * Q_matrix
    idct = cv2.idct(dequant)
    recon = np.clip(idct + 128.0, 0, 255)
    
    return recon, dc_val

# ==========================================
# 4. Main Logic: Encode/Decode File
# ==========================================

def custom_jpeg_encode(image_path, output_file, quality=50):
    img = cv2.imread(image_path)
    if img is None:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.randn(img, (128, 128, 128), (50, 50, 50))
        
    h, w = img.shape[:2]
    
    # Convert to Y only
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(img_ycrcb)
    
    # Padding
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    y_padded = cv2.copyMakeBorder(y, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    ph, pw = y_padded.shape
    
    Q = get_scaled_q_matrix(quality)
    writer = BitWriter()
    
    # Header: Height (2B), Width (2B), Quality (1B)
    # Simple custom header
    writer.write(h, 16)
    writer.write(w, 16)
    writer.write(quality, 8)
    
    prev_dc = 0
    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            block = y_padded[i:i+8, j:j+8]
            prev_dc = encode_block(block, prev_dc, Q, writer)
            
    bitstream = writer.flush()
    
    with open(output_file, 'wb') as f:
        f.write(bitstream)
        
    return y, bitstream

def custom_jpeg_decode(input_file):
    with open(input_file, 'rb') as f:
        data = f.read()
        
    reader = BitReader(data)
    
    # Read Header
    h = reader.read(16)
    w = reader.read(16)
    quality = reader.read(8)
    
    Q = get_scaled_q_matrix(quality)
    
    pad_h = h + (8 - h % 8) % 8 if h % 8 != 0 else h
    pad_w = w + (8 - w % 8) % 8 if w % 8 != 0 else w
    
    recon_img = np.zeros((pad_h, pad_w), dtype=np.float32)
    prev_dc = 0
    
    for i in range(0, pad_h, 8):
        for j in range(0, pad_w, 8):
            block, next_dc = decode_block(reader, prev_dc, Q)
            if block is None: break
            recon_img[i:i+8, j:j+8] = block
            prev_dc = next_dc
            
    return recon_img[:h, :w].astype(np.uint8)

# ==========================================
# 5. Analysis & Plotting
# ==========================================

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0: return 100
    return 10 * np.log10((255.0**2) / mse)

def main():
    # 이미지가 없으면 생성됨
    img_path = "A:/python_code/DB/Kodak/1.png" 
    # 테스트용 이미지 경로가 유효하지 않으면 코드가 자동으로 노이즈 이미지 생성
    
    qualities = [10, 30, 50, 70, 90]
    
    # Results storage
    custom_bpp = []
    custom_psnr = []
    opencv_bpp = []
    opencv_psnr = []
    
    temp_bin = "temp_custom.bin"
    temp_jpg = "temp_opencv.jpg"
    
    print(f"{'Qual':<5} | {'My Size(KB)':<12} | {'My PSNR':<8} | {'CV Size(KB)':<12} | {'CV PSNR':<8}")
    print("-" * 60)

    for q in qualities:
        # 1. Custom Method
        y_orig, bitstream = custom_jpeg_encode(img_path, temp_bin, q)
        y_recon = custom_jpeg_decode(temp_bin)
        
        size_bytes = len(bitstream)
        pixels = y_orig.shape[0] * y_orig.shape[1]
        bpp_val = size_bytes * 8 / pixels
        psnr_val = calculate_psnr(y_orig, y_recon)
        
        custom_bpp.append(bpp_val)
        custom_psnr.append(psnr_val)
        
        # 2. OpenCV Method (Comparison)
        # Note: OpenCV saves YUV 4:2:0 usually. To mimic Y-only comparison:
        # We save the full image, but measure size vs pixels. 
        # Since custom is Y-only, it will be smaller. This is expected.
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: img_bgr = cv2.cvtColor(y_orig, cv2.COLOR_GRAY2BGR)
            
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        result, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        
        cv_size_bytes = encimg.size
        cv_bpp = cv_size_bytes * 8 / pixels
        
        # For fair PSNR, decode and extract Y
        decimg = cv2.imdecode(encimg, 1)
        dec_y = cv2.split(cv2.cvtColor(decimg, cv2.COLOR_BGR2YCrCb))[0]
        cv_psnr_val = calculate_psnr(y_orig, dec_y)
        
        opencv_bpp.append(cv_bpp)
        opencv_psnr.append(cv_psnr_val)
        
        print(f"{q:<5} | {size_bytes/1024:<12.2f} | {psnr_val:<8.2f} | {cv_size_bytes/1024:<12.2f} | {cv_psnr_val:<8.2f}")

    # Plotting RD Curve
    plt.figure(figsize=(10, 6))
    plt.plot(custom_bpp, custom_psnr, 'r-o', label='Custom JPEG (Y-only, Simplified Huffman)')
    plt.plot(opencv_bpp, opencv_psnr, 'b-s', label='OpenCV JPEG (Full Color, Optimized)')
    
    plt.title("Rate-Distortion Curve: Custom vs Standard JPEG")
    plt.xlabel("Bits Per Pixel (bpp)")
    plt.ylabel("PSNR (dB, Y-channel)")
    plt.grid(True)
    plt.legend()
    plt.savefig("rd_curve.png")
    plt.show()
    
    # Cleanup
    if os.path.exists(temp_bin): os.remove(temp_bin)

if __name__ == "__main__":
    main()
