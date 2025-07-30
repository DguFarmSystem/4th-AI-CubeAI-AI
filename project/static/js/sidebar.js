// 좌측 대분류 탭 전환
document.querySelectorAll('.left-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.left-tab').forEach(t=>t.classList.remove('active'));
    tab.classList.add('active');
    const key = tab.dataset.tab;  // pre, model, train, eval
    document.querySelectorAll('.pane').forEach(p=>p.hidden = true);
    document.getElementById('pane-' + key).hidden = false;
  });
});


// 블록 활성/비활성 토글 함수
function setBlockActive(block, isActive) {
  block.dataset.active = isActive ? 'true' : 'false';
  block.querySelectorAll('input, select').forEach(el => {
    el.disabled = !isActive;
  });
}

// 초기 상태 설정
document.querySelectorAll('.pane').forEach(pane => {
  pane.querySelectorAll('.block').forEach(block => {
    const id = block.id;
    let active = false;
    switch(id) {
      case 'block-data-selection': active = true; break;
      case 'block-drop-na':       active = document.querySelector('input[name="drop_na"]').checked; break;
      case 'block-drop-bad':      active = document.querySelector('input[name="drop_bad"]').checked; break;
      case 'block-split-xy':      active = document.querySelector('input[name="split_xy"]').checked; break;
      case 'block-resize':        active = document.querySelector('input[name="resize_n"]').value !== ''; break;
      case 'block-augment':       active = document.querySelector('select[name="augment_method"]').value !== ''; break;
      case 'block-normalize':     active = document.querySelector('select[name="normalize"]').value !== ''; break;
      default:                    active = false;
    }
    setBlockActive(block, active);
  });
});

// 블록 클릭 시 토글 (단, 내부 컨트롤 클릭 제외)
document.querySelectorAll('.block').forEach(block => {
  block.addEventListener('click', e => {
    if (['INPUT','SELECT','LABEL'].includes(e.target.tagName)) return;
    setBlockActive(block, block.dataset.active === 'false');
  });
});

// 테스트 사용 여부에 따른 필드 토글
document.getElementById('is_test').onchange = function() {
  document.getElementById('test-div').style.display = this.value === 'true' ? 'block' : 'none';
  document.getElementById('ratio-div').style.display = this.value === 'false' ? 'block' : 'none';
};
document.getElementById('is_test').dispatchEvent(new Event('change'));

// 잘못된 라벨 파라미터 토글
document.getElementById('drop_bad').onchange = function() {
  document.getElementById('drop_bad_params').style.display = this.checked ? 'block' : 'none';
};
document.getElementById('drop_bad').dispatchEvent(new Event('change'));
