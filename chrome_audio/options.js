(function () {
  const requestMicBtn = document.getElementById('requestMicBtn');
  const micStatusEl = document.getElementById('micStatus');
  const closeHintEl = document.getElementById('closeHint');

  function setStatus(text) {
    micStatusEl.textContent = text;
  }

  async function requestMic() {
    setStatus('Статус: запрашиваю доступ к микрофону…');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop());
      setStatus('Статус: доступ к микрофону разрешен.');
      closeHintEl.classList.remove('hidden');
      return true;
    } catch (err) {
      const msg = err && err.message ? err.message : 'не удалось получить доступ';
      setStatus('Статус: ошибка — ' + msg);
      return false;
    }
  }

  requestMicBtn.addEventListener('click', () => {
    requestMic();
  });

  const params = new URLSearchParams(window.location.search);
  if (params.get('mic') === '1') {
    requestMic();
  }
})();
