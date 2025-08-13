// static/js/sidebar.js

function setBlockActive(block, active) {
  block.dataset.active = active ? 'true' : 'false';

  // 필수 블록은 항상 활성 상태(입력 disabled 금지)
  const isRequired = block.classList.contains('block-required');

  block.querySelectorAll('input, select, textarea, button').forEach(el => {
    if (isRequired) {
      el.disabled = false;
    } else {
      el.disabled = !active;
    }
  });

  // 시각 효과는 CSS에서 [data-active]로 처리
}

// 좌측 대분류(전처리/모델/학습/평가) 탭 전환
function initLeftTabs() {
  document.querySelectorAll('.left-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.left-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const key = tab.dataset.tab; // pre, model, train, eval
      document.querySelectorAll('.pane').forEach(p => p.hidden = true);
      document.getElementById('pane-' + key).hidden = false;
    });
  });

  // 초기 표시: 전처리 탭
  const first = document.querySelector('.left-tab[data-tab="pre"]');
  if (first) first.click();
}

// 블록 초기화: data-active / block-required 기반으로 통일
function initBlocks() {
  document.querySelectorAll('.block').forEach(block => {
    const isRequired = block.classList.contains('block-required');
    const wantActive = (block.dataset.active || 'false') === 'true';
    setBlockActive(block, isRequired || wantActive);
  });

  // 블록 클릭 시 토글 (내부 컨트롤 클릭은 제외, 필수 블록 제외)
  document.querySelectorAll('.block').forEach(block => {
    block.addEventListener('click', e => {
      const tag = e.target.tagName;
      if (['INPUT','SELECT','LABEL','OPTION','TEXTAREA','BUTTON'].includes(tag)) return;
      if (block.classList.contains('block-required')) return;
      const now = block.dataset.active === 'true';
      setBlockActive(block, !now);
    });
  });
}

// 의존 필드 토글(테스트 사용 여부, 라벨 필터)
function initDependentFields() {
  const isTest = document.getElementById('is_test');
  if (isTest) {
    const syncTest = () => {
      const yes = isTest.value === 'true';
      const testDiv  = document.getElementById('test-div');
      const ratioDiv = document.getElementById('ratio-div');
      if (testDiv)  testDiv.style.display  = yes ? 'block' : 'none';
      if (ratioDiv) ratioDiv.style.display = yes ? 'none' : 'block';
    };
    isTest.addEventListener('change', syncTest);
    syncTest();
  }

  const dropBad = document.getElementById('drop_bad');
  if (dropBad) {
    const syncBad = () => {
      const p = document.getElementById('drop_bad_params');
      if (p) p.style.display = dropBad.checked ? 'block' : 'none';
    };
    dropBad.addEventListener('change', syncBad);
    syncBad();
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initLeftTabs();
  initBlocks();
  initDependentFields();
});