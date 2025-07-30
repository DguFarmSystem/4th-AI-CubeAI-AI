import os
import io
import base64
from flask import Flask, render_template, request, send_from_directory, jsonify

import pandas as pd
import torch, numpy as np
from PIL import Image
from torchvision import transforms

from blocks.Preprocessing.data_selection     import generate_data_selection_snippet
from blocks.Preprocessing.drop_na           import generate_drop_na_snippet
from blocks.Preprocessing.drop_bad_labels   import generate_drop_bad_labels_snippet
from blocks.Preprocessing.split_xy          import generate_split_xy_snippet
from blocks.Preprocessing.resize            import generate_resize_snippet
from blocks.Preprocessing.augment           import generate_augment_snippet
from blocks.Preprocessing.normalize         import generate_normalize_snippet

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    snippet = ""
    # 현재 디렉토리의 모든 CSV 파일 목록 (원본 + 변환된)
    options = [f for f in os.listdir('.') if f.endswith('.csv')]

    if request.method == 'POST':
        # 1) 데이터 선택 블록 파라미터
        dataset      = request.form['dataset']
        is_test      = request.form['is_test']
        testdataset  = request.form.get('testdataset','')
        a            = request.form.get('a','100')

        # 2) 결측치 삭제 블록 파라미터
        drop_na_flag = 'drop_na' in request.form

        # 3) 잘못된 라벨 삭제 블록 파라미터
        drop_bad_flag = 'drop_bad' in request.form
        min_label    = request.form.get('min_label','0')
        max_label    = request.form.get('max_label','9')

        # 4) 입력/라벨 분리 블록 파라미터
        split_xy_flag = 'split_xy' in request.form

        # 5) 이미지 크기 변경 블록 파라미터
        resize_n     = request.form.get('resize_n')

        # 6) 이미지 증강 블록 파라미터
        augment_m    = request.form.get('augment_method')
        augment_p    = request.form.get('augment_param')

        # 7) 픽셀 값 정규화 블록 파라미터
        normalize_m  = request.form.get('normalize')

        # — 코드 스니펫 조립 순서 —
        lines = [
            "# 자동 생성된 show.py",
            "# 필요한 라이브러리 임포트",
            "import pandas as pd",
            "import torch, numpy as np",
            "from PIL import Image",
            "from torchvision import transforms",
            ""
        ]
        # 1) 데이터 불러오기
        lines.append(generate_data_selection_snippet(dataset, is_test, testdataset, a))
        lines.append("")
        # 2) 결측치 삭제
        if drop_na_flag:
            lines.append(generate_drop_na_snippet())
            lines.append("")
        # 3) 잘못된 라벨 삭제
        if drop_bad_flag:
            lines.append(generate_drop_bad_labels_snippet(min_label, max_label))
            lines.append("")
        # 4) 입력/라벨 분리
        if split_xy_flag:
            lines.append(generate_split_xy_snippet())
            lines.append("")
        # 5) 이미지 크기 변경
        if resize_n:
            lines.append(generate_resize_snippet(int(resize_n)))
            lines.append("")
        # 6) 이미지 증강
        if augment_m and augment_p:
            lines.append(generate_augment_snippet(augment_m, int(augment_p)))
            lines.append("")
        # 7) 픽셀 값 정규화
        if normalize_m:
            lines.append(generate_normalize_snippet(normalize_m))
            lines.append("")

        snippet = "\n".join(lines)

        # 1) show.py 파일 생성
        with open('show.py', 'w', encoding='utf-8') as f:
            f.write(snippet)

        # 2) exec를 통해 변환된 train_df/test_df 저장
        namespace = {}
        full_code = (
            "import pandas as pd\n"
            "import torch\n"
            "import numpy as np\n"
            "from PIL import Image\n"
            "from torchvision import transforms\n"
            + snippet
        )
        # globals와 locals를 같은 dict로 넘겨 줍니다.
        # exec(full_code, namespace, namespace)
        if 'train_df' in namespace:
            namespace['train_df'].to_csv('train_transformed.csv', index=False)
        if 'test_df' in namespace:
            namespace['test_df'].to_csv('test_transformed.csv', index=False)

        # 3) 옵션 갱신
        options = [f for f in os.listdir('.') if f.endswith('.csv')]

    return render_template('index.html', snippet=snippet, options=options)

@app.route('/download')
def download_file():
    return send_from_directory('.', 'show.py', as_attachment=True)

@app.route('/data-info')
def data_info():
    """
    ?file=...&type=shape|structure|sample|images&n=...
    """
    file = request.args.get('file')
    info_type = request.args.get('type','shape')
    n = int(request.args.get('n',5))
    df = pd.read_csv(file)

    if info_type == 'shape':
        return jsonify(rows=df.shape[0], cols=df.shape[1])
    if info_type == 'structure':
        cols = [{"name":c,"dtype":str(df[c].dtype)} for c in df.columns]
        return jsonify(columns=cols)
    if info_type == 'sample':
        sample = df.head(n).values.tolist()
        return jsonify(columns=list(df.columns), sample=sample)
    if info_type == 'images':
        images = []
        for _,row in df.head(n).iterrows():
            pix = row.values[1:].astype(int)
            arr = pix.reshape(28,28).astype('uint8')
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()
            images.append(b64)
        return jsonify(images=images)
    return jsonify({})

if __name__ == "__main__":
    # app.run(debug=True)
    # app.run(debug=True, use_reloader=False)

    # 로컬 호스트의 9000번 포트에서만 서비스
    app.run(host="127.0.0.1", port=9000, debug=True, use_reloader=False)

