# blocks/data_selection.py

def generate_data_selection_snippet(dataset, is_test, testdataset, a):
    """
    train_df / test_df 로드 및 분할 스니펫 생성
    """
    lines = []
    lines.append("# -----------------------------")
    lines.append("# ---------[데이터 선택 블록]---------")
    lines.append("# -----------------------------")
    lines.append(f"train_df = pd.read_csv('{dataset}')  # '{dataset}' 파일에서 학습용 데이터로드")
    lines.append("")
    if is_test == 'true':
        lines.append("test_df  = pd.read_csv('{testdataset}')  # 사용자가 지정한 테스트 데이터로드")
    else:
        lines.append(f"# 테스트 미지정 → 학습 데이터를 {a}% 사용, 나머지 {(100-int(a))}%를 테스트로 분할")
        lines.append(f"test_df  = train_df.sample(frac={(100-int(a))/100.0}, random_state=42)")
        lines.append("train_df = train_df.drop(test_df.index)")
    lines.append("")
    return "\n".join(lines)
