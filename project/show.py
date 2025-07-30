# 자동 생성된 show.py
# 필요한 라이브러리 임포트
import pandas as pd
import torch, numpy as np
from PIL import Image
from torchvision import transforms

# -----------------------------
# ---------[데이터 선택 블록]---------
# -----------------------------
train_df = pd.read_csv('mnist_test.csv')  # 'mnist_test.csv' 파일에서 학습용 데이터로드

test_df  = pd.read_csv('{testdataset}')  # 사용자가 지정한 테스트 데이터로드


# -----------------------------
# ------[빈 데이터 삭제 블록]------
# -----------------------------
train_df = train_df.dropna()  # 학습 데이터에서 NaN 포함 행 제거
test_df  = test_df.dropna()   # 테스트 데이터에서 NaN 포함 행 제거


# -----------------------------
# --[잘못된 라벨 삭제 블록]--
# -----------------------------
# 라벨값 허용 범위: 3 ~ 9
train_df = train_df[train_df['label'].between(3, 9)]  # 학습 데이터 필터링
test_df  = test_df[test_df['label'].between(3, 9)]   # 테스트 데이터 필터링


# -----------------------------
# ------[입력/라벨 분리 블록]------
# -----------------------------
import torch

# 1) 학습 데이터(X_train, y_train) 분리
X_train = train_df.iloc[:, 1:].values  # 학습용 입력 데이터 (NumPy 배열)
y_train = train_df.iloc[:, 0].values     # 학습용 라벨 데이터 (NumPy 배열)
y_train = torch.from_numpy(y_train).long()  # NumPy → LongTensor 변환

# 2) 테스트 데이터(X_test, y_test) 분리
X_test  = test_df.iloc[:, 1:].values   # 테스트용 입력 데이터 (NumPy 배열)
y_test  = test_df.iloc[:, 0].values     # 테스트용 라벨 데이터 (NumPy 배열)
y_test  = torch.from_numpy(y_test).long()   # NumPy → LongTensor 변환


# -----------------------------
# ----[이미지 크기 변경 블록]-----
# -----------------------------
import numpy as np
from torchvision import transforms

# 이미지 크기 변경: 28×28
# - X_train, X_test 은 현재 NumPy 배열(shape: N×784)
transform = transforms.Compose([
    transforms.ToPILImage(),              # NumPy 배열 → PIL 이미지
    transforms.Resize((28, 28)),  # 지정 크기로 리사이즈
    transforms.ToTensor()                 # PIL → Tensor (C×H×W), 값 0~1
])

# 1) 학습 데이터 리사이즈
images_2d = X_train.reshape(-1, 28, 28).astype(np.uint8)   # 1D→2D 전환
X_train = torch.stack([transform(img) for img in images_2d], dim=0)

# 2) 테스트 데이터 리사이즈
images_2d = X_test.reshape(-1, 28, 28).astype(np.uint8)
X_test  = torch.stack([transform(img) for img in images_2d], dim=0)


# -----------------------------
# ---[이미지 증강 블록]---
# -----------------------------
# 방법: vflip, 파라미터: 10
from torchvision import transforms
transform_aug = transforms.RandomVerticalFlip(p=1.0)    # 수직 뒤집기

# 학습 데이터 증강 및 라벨 복제
aug_train = torch.stack([transform_aug(x) for x in X_train], dim=0)  # 증강된 이미지
X_train = torch.cat([X_train, aug_train], dim=0)  # 원본+증강 이미지 합치기
y_train = torch.cat([y_train, y_train], dim=0)   # 라벨도 원본 복제하여 합치기

# 테스트 데이터 증강 및 라벨 복제
aug_test  = torch.stack([transform_aug(x) for x in X_test], dim=0)   # 증강된 이미지
X_test   = torch.cat([X_test, aug_test], dim=0)   # 원본+증강 이미지 합치기
y_test   = torch.cat([y_test, y_test], dim=0)     # 테스트 라벨 복제하여 합치기


# -----------------------------
# ---[픽셀 값 정규화 블록]---
# -----------------------------
# 0~1 범위로 스케일링
X_train = X_train / 255.0
X_test  = X_test  / 255.0

