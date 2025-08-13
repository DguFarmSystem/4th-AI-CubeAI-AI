# blocks/Training/start.py

from .loss            import generate_loss_snippet
from .optimizer       import generate_optimizer_snippet
from .training_option import generate_training_option_snippet

def generate_training_snippet(form):
    """
    form: request.form 딕셔너리
    → ⑤~⑦ 학습하기 단계 전체 스니펫 생성
    """
    # 1) form에서 파라미터 추출
    loss_method      = form.get('loss_method', '')
    optimizer_method = form.get('optimizer_method', '')
    learning_rate    = form.get('learning_rate', '')
    epochs           = form.get('epochs', '')
    batch_size       = form.get('batch_size', '')
    patience         = form.get('patience', '')

    lines = []
    # 학습 모듈 import
    lines.append("# -----------------------------")
    lines.append("# --------- [학습하기 블록] ---------")
    lines.append("# -----------------------------")
    lines.append("import torch.nn as nn")
    lines.append("import torch.optim as optim")
    lines.append("")

    # ⑤ 손실함수
    if loss_method:
        lines.append(generate_loss_snippet(loss_method))

    # ⑥ 옵티마이저
    if optimizer_method and learning_rate:
        lr = float(learning_rate)
        lines.append(generate_optimizer_snippet(optimizer_method, lr))

    # ⑦ 학습 옵션
    if epochs and batch_size and patience:
        e = int(epochs)
        bs = int(batch_size)
        p = int(patience)
        lines.append(generate_training_option_snippet(e, bs, p))

    return "\n".join(lines)
