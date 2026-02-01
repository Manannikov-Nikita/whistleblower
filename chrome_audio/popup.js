(function () {
  const recordBtn = document.getElementById('recordBtn');
  const statusEl = document.getElementById('status');
  const warningEl = document.getElementById('warning');
  const includeMicEl = document.getElementById('includeMic');
  const micPermissionEl = document.getElementById('micPermission');
  const micStatusEl = document.getElementById('micStatus');
  const requestMicBtn = document.getElementById('requestMicBtn');

  function setStatus(text, isRecording = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle('recording', isRecording);
  }

  function showRecordingUI() {
    recordBtn.textContent = '';
    recordBtn.innerHTML = '<span class="btn-icon">■</span> Остановить';
    recordBtn.classList.add('stop');
    recordBtn.disabled = false;
    warningEl.classList.remove('hidden');
    setStatus('Идёт запись… Окно можно закрыть', true);
  }

  function showIdleUI() {
    recordBtn.textContent = '';
    recordBtn.innerHTML = '<span class="btn-icon">●</span> Начать запись';
    recordBtn.classList.remove('stop');
    warningEl.classList.add('hidden');
    setStatus('Готов к записи', false);
  }

  function showMicPermission(text) {
    micStatusEl.textContent = text;
    micPermissionEl.classList.remove('hidden');
  }

  function hideMicPermission() {
    micPermissionEl.classList.add('hidden');
  }

  async function requestMicPermission() {
    setStatus('Запрос доступа к микрофону…', false);
    try {
      const micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micStream.getTracks().forEach((t) => t.stop());
      hideMicPermission();
      setStatus('Готов к записи', false);
      return true;
    } catch (err) {
      const msg = (err.message || '').toLowerCase();
      const isPermissionError =
        msg.includes('permission') ||
        msg.includes('dismissed') ||
        msg.includes('denied') ||
        msg.includes('notallowed') ||
        err.name === 'NotAllowedError';
      if (isPermissionError) {
        showMicPermission('Нет доступа к микрофону. Нажмите кнопку ниже и разрешите доступ.');
        setStatus('Микрофон не разрешен', false);
        return false;
      }
      showMicPermission('Ошибка микрофона: ' + (err.message || 'не удалось получить доступ'));
      setStatus('Микрофон: ошибка', false);
      return false;
    }
  }

  function updateUIFromState() {
    chrome.runtime.sendMessage({ type: 'getRecordingState' }, (res) => {
      if (chrome.runtime.lastError) {
        showIdleUI();
        return;
      }
      if (res && res.isRecording) {
        showRecordingUI();
      } else {
        showIdleUI();
      }
    });
  }

  requestMicBtn.addEventListener('click', async () => {
    await requestMicPermission();
  });

  includeMicEl.addEventListener('change', () => {
    if (!includeMicEl.checked) {
      hideMicPermission();
    }
  });

  recordBtn.addEventListener('click', async () => {
    chrome.runtime.sendMessage({ type: 'getRecordingState' }, async (stateRes) => {
      if (stateRes && stateRes.isRecording) {
        chrome.runtime.sendMessage({ type: 'stopRecording' });
        showIdleUI();
        return;
      }

      setStatus('Захват звука вкладки…', false);

      const [tab] = await new Promise((resolve) => {
        chrome.tabs.query({ active: true, currentWindow: true }, resolve);
      });
      if (!tab || !tab.id) {
        setStatus('Нет активной вкладки', false);
        return;
      }

      let streamId;
      try {
        streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
      } catch (e) {
        setStatus('Ошибка: ' + (e.message || 'не удалось получить звук вкладки'), false);
        return;
      }

      const includeMic = includeMicEl.checked;

      if (includeMic) {
        const ok = await requestMicPermission();
        if (!ok) {
          return;
        }
      }

      setStatus('Запуск записи в фоне…', false);

      chrome.runtime.sendMessage(
        { type: 'startRecording', streamId, includeMic },
        (res) => {
          if (chrome.runtime.lastError) {
            setStatus('Ошибка: ' + chrome.runtime.lastError.message, false);
            return;
          }
          if (res && !res.ok) {
            setStatus('Ошибка: ' + (res.error || 'не удалось запустить'), false);
            return;
          }
          showRecordingUI();
        }
      );
    });
  });

  updateUIFromState();
})();
