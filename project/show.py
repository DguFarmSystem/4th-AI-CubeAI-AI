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

# 테스트 미지정 → 학습 데이터를 80% 사용, 나머지 20%를 테스트로 분할
test_df  = train_df.sample(frac=0.2, random_state=42)
train_df = train_df.drop(test_df.index)


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



import torch.nn as nn
import torch.optim as optim

# -----------------------------
# --------- [모델설계 블록] ---------
# -----------------------------


# -----------------------------
# --------- [학습하기 블록] ---------
# -----------------------------
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# -------[학습 옵션 블록]--------
# -----------------------------
# epochs=10, batch_size=64, patience=3
num_epochs = 10        # 전체 학습 반복 횟수
batch_size = 64     # 한 배치 크기
patience = 3         # 조기 종료 전 대기 에폭 수


# -----------------------------
# --------- [평가하기 블록] ---------
# -----------------------------
