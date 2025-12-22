#include <algorithm> // for std::clamp
#include <cmath>     // for std::abs
#include <bit>       // C++20 for std::bit_width (없으면 아래 대체 코드 사용)

/**
 * @param value      인코딩할 실제 계수 값
 * @param probs      Python 결과 확률 리스트 (0~65536 스케일)
 * @param probCount  확률 리스트 길이 (반드시 2 이상이어야 함: P(0) + P(Esc))
 */
void CABACWriter::encodeSequentialBinary(int value, const uint16_t* probs, int probCount)
{
  // 방어 코드: 확률 테이블이 최소한 (값, Escape) 구성을 가져야 함
  // 실제 릴리즈 빌드에서는 제거 가능
  // assert(probCount >= 2); 

  uint32_t absVal = (uint32_t)std::abs(value); // abs는 int 반환이므로 캐스팅

  // 비교할 Unary Bin의 개수 (마지막 Escape 확률은 암묵적 1 처리하므로 제외)
  int explicitLimit = probCount - 1;

  // -------------------------------------------------------
  // 1. Context Coded Bins (Truncated Unary)
  // -------------------------------------------------------
  for (int k = 0; k < explicitLimit; ++k)
  {
    // [Safety] 확률 값이 0이나 65536이 되면 산술코덱이 깨질 수 있음.
    // 최소 1, 최대 65535로 안전하게 클램핑 (Python에서 보정했더라도 이중 안전장치 권장)
    uint32_t prob = std::clamp<uint32_t>(probs[k], 1, 65535);

    if (absVal == k)
    {
      // P(Stop) = prob. Symbol '1' means Stop.
      encodeManual(1, prob);
      return; // 인코딩 완료
    }
    else
    {
      // P(Continue) = 1 - prob. Symbol '0' means Continue.
      encodeManual(0, prob);

      // [Sign Bit] 0이 아님이 확정된 직후 부호 전송 (Bypass)
      if (k == 0) 
      {
        m_binEncoder.encodeBinEP(value < 0 ? 1 : 0);
      }
    }
  }

  // -------------------------------------------------------
  // 2. Escape Coding (Exp-Golomb 0th order)
  // -------------------------------------------------------
  // 여기까지 왔다면 absVal >= explicitLimit 임이 확정됨 (Implicit Escape)
  
  uint32_t rem = absVal - explicitLimit;
  uint32_t val = rem + 1;

  // [Optimization] 고속 Log2 계산 (Floor Log2)
  // C++20: int log2V = std::bit_width(val) - 1;
  // GCC/Clang: int log2V = 31 - __builtin_clz(val);
  // MSVC: _BitScanReverse 이용
  
  // (호환성을 위한 기존 로직 유지하되, 주석으로 최적화 포인트 명시)
  int log2V = 0;
  uint32_t temp = val;
  while (temp >>= 1) { log2V++; } 

  // 1) Prefix: Unary part (log2V 개수만큼 '0')
  for (int i = 0; i < log2V; i++)
  {
    m_binEncoder.encodeBinEP(0);
  }

  // 2) Suffix: Binary part ('1' + remainder bits)
  // val의 유효 비트 수는 (log2V + 1)개입니다.
  // 최상위 비트는 항상 1이며(Separator), 그 뒤에 나머지 비트가 따라옵니다.
  m_binEncoder.encodeBinsEP(val, log2V + 1);
}
