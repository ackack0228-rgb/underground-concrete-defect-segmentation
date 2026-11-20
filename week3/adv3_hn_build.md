# 심화 ③ 하드네거티브 수집

- train 영상 3918장 → 라벨과 겹치지 않는 조각 후보 8481개 (영상의 2/3 크기, 3×3 격자, 라벨 +16px 여유)
- 기준 모델이 검증 임계값 이상으로 예측한 조각(하드네거티브) 2190개 → 원본 ID당 최대 2장 · 최대 200장 = **108장** (원본 ID 56개)
- 조각이 유발한 오탐 클래스: crack 31 · efflorescence 36 · spall_detach 41
- 새 데이터: `~\datasets\crack-seg-hn` (train 3918 + 108 배경 = 4026장, val · test 그대로)
- 평가 전용 음성 조각(선택 · 학습에 안 씀): val 88개 · test 59개 → `~\datasets\crack-seg-neg-eval`
