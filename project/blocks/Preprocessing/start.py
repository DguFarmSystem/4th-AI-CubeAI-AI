# blocks/Preprocessing/start.py

from .data_selection     import generate_data_selection_snippet
from .drop_na           import generate_drop_na_snippet
from .drop_bad_labels   import generate_drop_bad_labels_snippet
from .split_xy          import generate_split_xy_snippet
from .resize            import generate_resize_snippet
from .augment           import generate_augment_snippet
from .normalize         import generate_normalize_snippet

def generate_preprocessing_snippet(form):
    """
    form: request.form 딕셔너리
    → preprocessing 단계(1~7) 전체 코드를 조립하여 반환
    """
    # 1) form에서 파라미터 꺼내기
    dataset      = form['dataset']
    is_test      = form['is_test']
    testdataset  = form.get('testdataset','')
    a            = form.get('a','100')
    drop_na_flag = 'drop_na' in form
    drop_bad_flag= 'drop_bad' in form
    min_label    = form.get('min_label','0')
    max_label    = form.get('max_label','9')
    split_xy_flag= 'split_xy' in form
    resize_n     = form.get('resize_n','')
    augment_m    = form.get('augment_method','')
    augment_p    = form.get('augment_param','')
    normalize_m  = form.get('normalize','')

    # 2) 스니펫 조립
    lines = [
        "# 자동 생성된 show.py",
        "# 필요한 라이브러리 임포트",
        "import pandas as pd",
        "import torch, numpy as np",
        "from PIL import Image",
        "from torchvision import transforms",
        "",
        # 1) 데이터 불러오기
        generate_data_selection_snippet(dataset, is_test, testdataset, a),
        ""
    ]

    # 2)–7) 블록별 조건적 삽입
    if drop_na_flag:
        lines += [ generate_drop_na_snippet(), "" ]
    if drop_bad_flag:
        lines += [ generate_drop_bad_labels_snippet(min_label, max_label), "" ]
    if split_xy_flag:
        lines += [ generate_split_xy_snippet(), "" ]
    if resize_n:
        lines += [ generate_resize_snippet(int(resize_n)), "" ]
    if augment_m and augment_p:
        lines += [ generate_augment_snippet(augment_m, int(augment_p)), "" ]
    if normalize_m:
        lines += [ generate_normalize_snippet(normalize_m), "" ]

    # 최종 문자열 반환
    return "\n".join(lines)
