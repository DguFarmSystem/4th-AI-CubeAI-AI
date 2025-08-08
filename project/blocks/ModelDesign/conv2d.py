# conv2d.py

from dataclasses import dataclass

@dataclass
class Conv2DLayer:
    # 입력층 설정 구조체 (첫 번째 계층에만 필요)
    in_channels: int  # 입력 채널 수 (1: 흑백, 3: 컬러)
    # 이미지 특징 찾기 (Conv2D) 구조체
    out_channels: int  
    kernel_size: int  
    padding: int  

def generate_conv2d_code(conv2d_block: Conv2DLayer) -> str:
    """
    Convolution 2D 블록 생성
    """
    lines = []
    lines.append("\t\t\t" + "nn.Conv2d(                                     # 두 번째 합성곱")
    lines.append(f"\t\t\t\t" + f"in_channels={conv2d_block.in_channels},                            # 이전 출력 채널 32")
    lines.append(f"\t\t\t\t" + f"out_channels={conv2d_block.out_channels},                           # 이번 출력 채널 64")
    lines.append(f"\t\t\t\t" + f"kernel_size={conv2d_block.kernel_size},                             # 커널 크기 3x3")
    lines.append(f"\t\t\t\t" + f"padding={conv2d_block.padding}                                  # 패딩 1")
    lines.append("\t\t\t\t" + "),")
    
    return "\n".join(lines) 