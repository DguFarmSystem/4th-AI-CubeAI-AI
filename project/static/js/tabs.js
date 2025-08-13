// 1) 오른쪽 상단 탭: 코드 vs 데이터 구조 (이미 있다면 그대로)
function showTab(tab) {
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('content-' + tab).classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}
document.getElementById('tab-code').onclick = () => showTab('code');
document.getElementById('tab-data').onclick = () => showTab('data');

// 2) 코드 영역의 단계별 서브 탭: Pre/Model/Train/Eval
document.querySelectorAll('.stage-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.stage-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const key = tab.dataset.stage; // pre, model, train, eval
    document.querySelectorAll('.stage-pane').forEach(p => p.hidden = true);
    document.getElementById('pane-code-' + key).hidden = false;
  });
});

// 3) 초기 상태: "코드" 탭 + "Preprocessing" 서브탭을 기본 표시
//   (페이지 처음 열거나 변환 직후에도 항상 보이도록)
document.getElementById('tab-code').click();
const firstStageTab = document.querySelector('.stage-tab[data-stage="pre"]');
if (firstStageTab) firstStageTab.click();
