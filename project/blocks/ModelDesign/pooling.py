# conv2d.py
from dataclasses import dataclass

@dataclass
class PoolingLayer:
    # 특징 크기 줄이기 (Pooling) 구조체
    pool_type: str  # Max 또는 Avg
    size: int       # 풀링 크기

def generate_pooling_code(pool_block: PoolingLayer) -> str:
    """
    Convolution 2D 블록 생성
    """
    lines = []
    lines.append(f"\t\t\t" + f"nn.{pool_block.pool_type}Pool2d({pool_block.size}, {pool_block.size}),                            # 풀링")
    
    return "\n".join(lines) 