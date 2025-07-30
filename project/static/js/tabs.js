function showTab(tab) {
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  document.getElementById('content-' + tab).classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
}
document.getElementById('tab-code').onclick = () => showTab('code');
document.getElementById('tab-data').onclick = () => showTab('data');
