# 블록코딩 AI 파이프라인

> **완전 초보자도 손쉽게 CNN 기반 손글씨 분류 파이프라인을 “블록” 형태로 조립할 수 있는 웹앱**
> 각 단계별로 “데이터 선택 → 전처리 → 모델 설계 → 학습 → 평가” 블록을 클릭·드래그하듯 활성화/비활성화하고,
> 한 번의 “변환하기”로 완전한 Python 스크립트(`show.py`)를 자동 생성합니다.

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [주요 기능](#주요-기능)
3. [환경 및 요구사항](#환경-및-요구사항)
4. [설치 및 초기 설정](#설치-및-초기-설정)
5. [프로젝트 구조](#프로젝트-구조)
6. [블록 모듈 세부 설명](#블록-모듈-세부-설명)
7. [로컬 실행 방법](#로컬-실행-방법)
8. [공개 URL로 노출하기 (역방향 SSH)](#공개-url로-노출하기-역방향-ssh)
9. [자주 묻는 질문](#자주-묻는-질문)
10. [라이선스](#라이선스)

---

## 프로젝트 개요

이 프로젝트는 **스크래치(Scratch) 스타일의 블록 코딩** 인터페이스를 통해
CNN 기반 손글씨(MNIST) 분류 파이프라인을 단계별로 구성하고,
이를 즉시 실행 가능한 Python (`show.py`) 코드로 변환해 줍니다.

* **목표 사용자**: AI/ML 완전 초보자, 교육용
* **핵심 아이디어**:

  1. 각 처리 단계(데이터 로드/전처리/모델 설계/학습/검증)를 “블록”으로 추상화
  2. 드래그 앤 클릭으로 단계 활성화 → 전체 스크립트 자동 생성
  3. Flask 기반 웹 UI + 역방향 SSH 터널로 언제 어디서나 접속 가능

---

## 주요 기능

* **데이터 선택 & 분할**

  * 로컬 또는 업로드된 CSV 파일 자동 감지
  * 훈련용/테스트용 분할 유무 선택
* **데이터 전처리(Preprocessing)**

  1. 빈 데이터(NaN) 제거
  2. 잘못된 라벨 필터링
  3. 입력/라벨 분리 → `X_train`, `y_train`, `X_test`, `y_test` 생성
  4. 이미지 리사이즈 (n×n)
  5. 이미지 증강 (회전·뒤집기·이동)
  6. 픽셀 값 정규화 (0~~1 또는 –1~~1)
* **모델 설계(Model Design)** *(준비 중…)*
* **학습하기(Training)** *(준비 중…)*
* **평가하기(Evaluation)** *(준비 중…)*
* **코드 탭**: 완성된 `show.py` 코드 미리보기 + 다운로드
* **데이터 구조 탭**: CSV 행/열 수, 컬럼 타입, 샘플 테이블, 이미지 샘플

---

## Farm AI 002 서버
* **노션 참조**
---

## 프로젝트 구조

```
project/
├── app.py
├── requirements.txt
├── show.py                ← 자동 생성되는 스크립트 (매번 덮어쓰기)
├── blocks/                ← 블록 모듈
│   ├── Preprocessing/     ← 데이터 전처리 블록 7종
│   ├── ModelDesign/       ← (추후) 모델 설계 블록
│   ├── Training/          ← (추후) 학습 블록
│   └── Evaluation/        ← (추후) 평가 블록
├── templates/
│   ├── layout.html        ← 공통 레이아웃
│   ├── sidebar.html       ← 좌측 블록 UI
│   ├── main_code.html     ← 코드 탭
│   └── main_data.html     ← 데이터 구조 탭
└── static/
    ├── css/
    │   └── style.css
    └── js/
        ├── sidebar.js
        ├── tabs.js
        └── data_info.js
```

---

## 블록 모듈 세부 설명

### Preprocessing (blocks/Preprocessing)

| 파일명                  | 역할                                                               |
| -------------------- | ---------------------------------------------------------------- |
| `data_selection.py`  | CSV 로드 & train/test 분할 스니펫 생성                                    |
| `drop_na.py`         | 결측치(NaN) 포함 행 제거 스니펫 생성                                          |
| `drop_bad_labels.py` | 라벨 범위(min,max) 벗어나는 행 제거 스니펫 생성                                  |
| `split_xy.py`        | DataFrame → `X_train`, `y_train`, `X_test`, `y_test` → Tensor 변환 |
| `resize.py`          | `(N×784)` → `(C×n×n)` 텐서 리사이즈 스니펫 생성                             |
| `augment.py`         | 회전·뒤집기·이동 증강 및 `X`,`y` 복제 & 병합 스니펫 생성                            |
| `normalize.py`       | 픽셀 값 정규화(0~~1 / -1~~1) 스니펫 생성                                    |

*(ModelDesign / Training / Evaluation 폴더는 추후 확장 예정)*

---

## 로컬 실행 방법

1. **Flask 호스트·포트 설정**
   `app.py` 맨 아래:

   ```python
   if __name__ == "__main__":
       # 로컬 전용: 127.0.0.1:9000
       app.run(host="127.0.0.1", port=9000, debug=True, use_reloader=False)
   ```

2. **서버 시작**

   ```bash
   python app.py
   ```

   * 출력: `* Running on http://127.0.0.1:9000/`

3. **브라우저 접속**

   ```
   http://127.0.0.1:9000
   ```

---

## 공개 URL로 노출하기 (역방향 SSH)

### 1. 로컬(Farm server)에서 리버스 터널 열기

```bash
# 터널 생성 (백그라운드 X, 대화형 유지)
ssh -N -R 0.0.0.0:9022:127.0.0.1:9000 root@211.188.56.255
```

* 네이버 클라우드 `0.0.0.0:9022` → 내 로컬 `127.0.0.1:9000`
---

## 자주 묻는 질문

**Q. `BadRequestKeyError: 'dataset'` 에러가 나는 이유는?**

* “데이터 선택” 블록(`dataset` 필드)은 **항상 필수**이므로 비활성화할 수 없습니다.
* `.block-required` 클래스로 고정되어 있으며, 삭제하거나 이름 변경해서는 안 됩니다.

**Q. 새로운 블록을 추가하려면?**

1. `blocks/ModelDesign/` 등 해당 폴더에 `*.py` 스니펫 함수 추가
2. `app.py` 에서 import & 조립 순서에 삽입
3. `sidebar.html`, `style.css`, `sidebar.js` 에 UI·스타일·동작 추가

---

## 라이선스

Apache 2.0 License © 2025 choconaena
