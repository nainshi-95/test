def convert_cdf_to_binary_probs(cdfTable, cdfLength, cdfOffset):
    """
    cdfTable: 각 행이 CDF인 이중 리스트
    cdfLength: 각 CDF의 길이 리스트
    cdfOffset: P(x<0)이 위치한 인덱스 (int 또는 list)
    
    Returns:
        binary_prob_list: 각 cdf에 대한 이진 조건부 확률 리스트들의 리스트
    """
    binary_prob_list = []

    for i in range(len(cdfTable)):
        cdf = cdfTable[i]
        length = cdfLength[i]
        
        # cdfOffset이 단일 int인지 리스트인지 확인하여 처리
        offset = cdfOffset[i] if isinstance(cdfOffset, list) else cdfOffset
        
        current_probs = []
        
        # 조건부 확률 계산을 위한 현재 남은 확률 질량 (초기값 1.0)
        remaining_mass = 1.0
        
        # 1. 0일 확률 (P(X=0))
        # CDF 정의상 P(X=0) = P(X<=0) - P(X<0)
        p_zero = cdf[offset + 1] - cdf[offset]
        
        # 0인지에 대한 Binary 확률 저장 (P(1) = Is Zero)
        # 조건부 확률: P(X=0 | Valid) = P(X=0) / remaining_mass
        if remaining_mass > 0:
            prob_is_zero = p_zero / remaining_mass
            current_probs.append(prob_is_zero)
            remaining_mass -= p_zero # 남은 확률 갱신 (0이 아닐 확률)
        else:
            current_probs.append(0.0)

        # 2. 양수 부분 확률 (P(|X|=k))
        # 대칭이므로 P(|X|=k) = 2 * P(X=k)
        # 마지막 요소(Tail Mass) 바로 전까지 루프
        # CDF 유효 데이터는 index 0 ~ length-2 까지 (length-1은 tail mass)
        
        k = 1
        # (offset + k + 1)이 Tail Mass 인덱스(length - 1)보다 작을 때까지
        while (offset + k + 1) < (length - 1):
            # 단측 확률: P(X=k) = P(X<=k) - P(X<k)
            # CDF 인덱스: P(X<=k)는 offset + k + 1, P(X<k)는 offset + k
            p_k_one_side = cdf[offset + k + 1] - cdf[offset + k]
            
            # 양측 확률 (대칭성 적용)
            p_k_abs = p_k_one_side * 2
            
            # 조건부 확률: P(|X|=k | |X| >= k)
            if remaining_mass > 0:
                prob_is_k = p_k_abs / remaining_mass
                current_probs.append(prob_is_k)
                remaining_mass -= p_k_abs
            else:
                current_probs.append(0.0)
            
            k += 1

        # 3. Escape 확률 (Tail Mass)
        # 마지막 요소는 예외(Escape) 확률
        p_escape = cdf[length - 1]
        
        # 이론적으로 마지막 단계에서 남은 확률(remaining_mass)은 p_escape와 같아야 함
        # 따라서 P(Escape | not previous) = 1.0 이 되어야 정상이지만,
        # 부동소수점 오차 등을 고려하여 계산식 유지
        if remaining_mass > 0:
            prob_is_escape = p_escape / remaining_mass
            # 보통 마지막은 무조건 Escape이므로 확률 1.0으로 클램핑하기도 함
            current_probs.append(min(prob_is_escape, 1.0))
        else:
            current_probs.append(1.0) # 남은게 없으면 강제 1.0 (혹은 에러처리)

        binary_prob_list.append(current_probs)

    return binary_prob_list

# --- 사용 예시 ---
# 가상의 CDF 데이터 (대칭형, 마지막은 Tail Mass)
# offset이 2라고 가정하면:
# idx 0: P(x<-1), idx 1: P(x<0), idx 2: P(x<0)[offset], idx 3: P(x<=0), idx 4: P(x<=1), idx 5: Tail
mock_cdf_table = [
    [0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.05], # 예시 1
    [0.1, 0.3, 0.5, 0.7, 0.9, 0.1]          # 예시 2 (길이가 다름)
]
mock_cdf_length = [7, 6]
mock_cdf_offset = 3 # P(x<0)의 인덱스라고 가정

# 오프셋 설명에 따라 cdf[offset] = P(x<0).
# 예시 1에서 offset=3 이면: cdf[3]=0.6 ?? -> 이러면 P(x<0)>0.5가 되어 대칭 붕괴.
# 대칭이려면 cdf[offset]은 보통 0.5 미만이어야 함. 
# 위 코드는 로직 검증용이므로, 실제 데이터에서는 올바른 offset 값을 넣으시면 됩니다.

# 실행
result = convert_cdf_to_binary_probs(mock_cdf_table, mock_cdf_length, 2)

print("변환된 Binary 확률 리스트:")
for idx, probs in enumerate(result):
    print(f"List {idx}: {probs}")
