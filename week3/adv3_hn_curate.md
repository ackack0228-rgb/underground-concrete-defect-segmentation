# 심화 ③ 하드네거티브 검토 (2차)

- 마이닝 108장을 조각마다 예측 상자를 그려 눈으로 판정: 남김 15 · 뺌(실제 결함) 13 · 뺌(판단 어려움) 80
- ★ 기준 모델이 '라벨 없는 곳'에서 높은 신뢰도로 찾은 것 중 적어도 13장은 라벨이 빠진 실제 결함(거북등 · 망상 균열, 박락 덩어리)이었다 → 자동 마이닝만으로 배경을 만들면 진짜 결함을 배경으로 가르치게 된다. 주석 누락 점검 도구로도 쓸 수 있다(주석기준서 v0.3 ⑤).
- 남긴 15장 + 1주차 직접 확인 3건 = 18개 영역 × 3번(오버샘플링) = 배경 영상 54장 → train 3972장
- 남긴 조각의 원인: 금속 고정구 · 배관 → spall_detach / 난간 그림자 → spall_detach / 배관 상단 브래킷 → spall_detach / 보수흔적(녹색 보수재) → crack (1주차) / 분필 곡선 → crack / 분필 곡선 → crack · efflorescence / 분필 표기 → efflorescence / 세로 배관 → crack · spall_detach / 앵커 구멍 · 케이블 → efflorescence / 앵커 구멍 → efflorescence / 이음부 · 금속 사다리 모서리 → spall_detach / 케이블 · 배관 → crack (1주차) / 케이블 고정 볼트 → spall_detach / 케이블 고정구 → spall_detach / 케이블 클램프 → spall_detach / 표면오염(얼룩) → efflorescence (1주차)
- 전체 판정표: adv3_hn_review.csv
