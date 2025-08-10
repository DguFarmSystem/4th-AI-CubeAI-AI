# blocks/Modeldesign/start.py

from .activation      import get_activation_function
from .cnn_create      import generate_cnn_class_code
from .conv2d          import generate_conv2d_code
from .dropout         import generate_dropout_code
from .fc_final        import generate_fc_final_code
from .fc_layer        import generate_fc_layer_code
from .pooling         import generate_pooling_code

def generate_modeldesign_snippet(form):

    # form에서 파라미터 호출
    in_channels = form.get('in_channels', '')
    out_channels = form.get('out_channels', '')
    kernel_size = form.get('kernel_size', '')
    padding = form.get('padding', '')
    p = form.get('p', '')
    activation_type = form.get('activation_type', '')
    pool_type = form.get('pool_type', '')
    size = form.get('size', '')
    dense_input_size = form.get('dense_input_size', '')
    dense_output_size = form.get('dense_output_size', '')
    num_classes = form.get('num_classes', '')

    lines = [
        "import torch.nn as nn",
        "import torch.optim as optim",
        ""
    ]
    # 학습 모듈 import
    lines.append("# -----------------------------")
    lines.append("# --------- [모델설계 블록] ---------")
    lines.append("# -----------------------------")
    lines.append("")


    return "\n".join(lines)
