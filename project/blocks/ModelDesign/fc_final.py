# conv2d.py
from dataclasses import dataclass

@dataclass
class OutputLayer:
    # 정답 예측층 (Output) 구조체
    num_classes: int  # 클래스 수
    dense_output_size: int

def generate_fc_final_code(output_block: OutputLayer) -> str:
    """
    Convolution 2D 블록 생성
    """
    lines = []
    lines.append(f"\t\t\t" + f"nn.Linear({output_block.dense_output_size}, {output_block.num_classes})                             # 최종 FC")
    lines.append("\t\t" + ")")

    return "\n".join(lines) 