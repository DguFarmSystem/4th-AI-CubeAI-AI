import os
import io
import base64
from flask import Flask, render_template, request, send_from_directory, jsonify

import pandas as pd
import torch, numpy as np
from PIL import Image
from torchvision import transforms

from blocks.Preprocessing.start import generate_preprocessing_snippet
from blocks.ModelDesign.start      import generate_modeldesign_snippet
from blocks.Training.start         import generate_training_snippet
from blocks.Evaluation.start       import generate_evaluation_snippet

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    # 단계별 코드 스니펫 초기화
    snippet_pre   = ""
    snippet_model = ""
    snippet_train = ""
    snippet_eval  = ""

    # CSV 옵션 읽기
    options = [f for f in os.listdir('.') if f.endswith('.csv')]

    if request.method == 'POST':
        try:
            # 1) Preprocessing
            snippet_pre   = generate_preprocessing_snippet(request.form)
            # 2) ModelDesign
            snippet_model = generate_modeldesign_snippet(request.form)
            # 3) Training
            snippet_train = generate_training_snippet(request.form)
            # 4) Evaluation
            snippet_eval  = generate_evaluation_snippet(request.form)

            # 전체 show.py 생성
            full = "\n\n".join(filter(None, [
                snippet_pre, snippet_model, snippet_train, snippet_eval
            ]))
            with open('show.py', 'w', encoding='utf-8') as f:
                f.write(full)

            # (선택) 변환된 DF 저장 — 현재 exec 비활성화 상태
            namespace = {}
            full_code = (
                "import pandas as pd\n"
                "import torch\n"
                "import numpy as np\n"
                "from PIL import Image\n"
                "from torchvision import transforms\n"
                + snippet_pre
            )
            # exec(full_code, namespace, namespace)  # 필요 시 주석 해제

            if 'train_df' in namespace:
                namespace['train_df'].to_csv('train_transformed.csv', index=False)
            if 'test_df' in namespace:
                namespace['test_df'].to_csv('test_transformed.csv', index=False)

            # 옵션 갱신 (새로 생긴 CSV 포함)
            options = [f for f in os.listdir('.') if f.endswith('.csv')]

        except Exception as e:
            # 서버 콘솔에 에러 출력 (템플릿은 정상 렌더)
            print(f"[ERROR] POST / index(): {e}", flush=True)

    # ✅ GET이든 POST든 항상 반환!
    return render_template(
        'index.html',
        snippet_pre=snippet_pre,
        snippet_model=snippet_model,
        snippet_train=snippet_train,
        snippet_eval=snippet_eval,
        options=options
    )

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
    # app.run(host="127.0.0.1", port=9000, debug=True, use_reloader=False)
    app.run(host="127.0.0.1", port=9099, debug=True, use_reloader=False)