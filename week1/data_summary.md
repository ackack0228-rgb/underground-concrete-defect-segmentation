# 데이터 요약 (제출물 ②)

- 경로: `~/datasets/crack-seg/images` (검색 패턴 `**/*` 재귀)
- 영상 수: **4116장**
- 열리지 않는 영상: 63장

## 해상도

| 가로 × 세로 | 영상 수 |
|---|---|
| 1920 × 1080 | 3727 |
| 1080 × 1920 | 326 |

- 열리는 4053장 기준 (열리지 않는 63장 제외).
- 원본은 모두 가로 영상이고, 세로 영상은 원본을 90° 회전한 증강본(파일명 `rot90`)이다.

## 데이터 구성 (폴더별)

| 폴더 | 영상 수 |
|---|---|
| test | 40 |
| train | 4036 |
| val | 40 |

## 데이터 구성 (원본 · 증강본)

| 폴더 | 원본 | 증강본 |
|---|---|---|
| test | 40 | 0 |
| train | 120 | 3916 |
| val | 40 | 0 |

- 파일명에 `_aug` · `_efflo_`가 붙은 영상은 원본을 회전 · 반전 · 밝기 · 대비 · 노이즈로 바꾼 증강본이다. val · test는 원본만 있다.

## 확장자

| 확장자 | 영상 수 |
|---|---|
| .jpg | 4116 |

## 데이터 출처

- 교수님 제공 실습용 데이터셋 (지하 콘크리트 구조물 표면 영상, 기준 논문과 같은 4,116장 구성).
- 기준 논문은 AI Hub 공개 데이터에서 고른 4,116장을 썼다. 추가 프로젝트에서는 AI Hub나 다른 공개 데이터셋을 쓸 수 있다 (강의 10쪽).

## 열리지 않는 영상

- 받은 압축 파일 안에 크기가 0바이트인 영상이 63장 있다 (다운로드 오류가 아니라 원본 압축 파일 상태). 이후 단계에서는 제외한다.
- 모두 train이고, 원본 1장(`train/URT020_2021_04_03_001_2914.jpg`)과 나머지 증강본이다.

- train/URT020_2021_01_12_002_779_aug_0_00944.jpg
- train/URT020_2021_01_12_002_779_augfill_00390_contrast.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00348_augfill_00785_brightness.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00362.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00362_augfill_00814_noise.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00362_augfill_01553_rot180_contrast.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00436.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00445.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00445_augfill_00221_hflip.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00483.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00483_augfill_00455_contrast.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00649.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00671.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00717.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00717_augfill_00350_contrast.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00730.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00730_augfill_01550_rot90ccw.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00853.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00853_augfill_01036_contrast.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00857.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00857_augfill_01389_vflip.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00875.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00933.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00933_augfill_00521_hflip_brightness.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00933_augfill_01386_rot180.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00960.jpg
- train/URT020_2021_03_05_002_20650_aug_3_00960_augfill_00846_rot90cw.jpg
- train/URT020_2021_03_05_002_20650_augfill_00756_rot180.jpg
- train/URT020_2021_03_05_002_20650_augfill_01308_vflip.jpg
- train/URT020_2021_04_03_001_2914.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00017.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00017_augfill_00469_rot180.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00017_augfill_01242_vflip.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00184.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00197.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00197_augfill_00744_rot180.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00221.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00221_augfill_01169_brightness.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00221_augfill_01474_rot180_contrast.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00327.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00327_augfill_00071_hflip_brightness.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00390.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00390_augfill_00072_rot90cw.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00522.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00522_augfill_00491_brightness.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00547.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00547_augfill_01395_rot180.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00555.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00604.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00604_augfill_00295_hflip_brightness.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00615.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00615_augfill_00330_rot90cw.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00682.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00835.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00835_augfill_01134_vflip.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00836.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00836_augfill_00004_contrast.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00859.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00909.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00918.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00935.jpg
- train/URT020_2021_04_03_001_2914_aug_0_00935_augfill_00589_hflip.jpg
- train/URT020_2021_04_03_001_2914_augfill_00384_rot180_contrast.jpg
