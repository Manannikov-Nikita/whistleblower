const OFFSCREEN_URL = 'offscreen.html';
const NATIVE_HOST_NAME = 'com.whistleblower.native_host';
let isRecording = false;
let lastError = null;
let nativePort = null;

async function ensureOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)],
  });
  if (contexts.length > 0) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_URL,
    reasons: ['USER_MEDIA'],
    justification: 'Запись звука вкладки и микрофона в фоне',
  });
}

async function closeOffscreenDocument() {
  try {
    await chrome.offscreen.closeDocument();
  } catch (_) {}
}

function setLastError(message) {
  lastError = message || null;
}

function ensureNativePort() {
  if (nativePort) return nativePort;
  try {
    nativePort = chrome.runtime.connectNative(NATIVE_HOST_NAME);
  } catch (err) {
    setLastError('Native host недоступен: ' + err.message);
    nativePort = null;
    return null;
  }

  nativePort.onMessage.addListener((msg) => {
    if (msg && msg.ok === false) {
      const detail = msg.error || 'неизвестная ошибка';
      setLastError('Native host: ' + detail);
    }
  });

  nativePort.onDisconnect.addListener(() => {
    const err = chrome.runtime.lastError;
    if (err) {
      setLastError('Native host недоступен: ' + err.message);
    } else {
      setLastError('Native host отключился.');
    }
    nativePort = null;
  });

  return nativePort;
}

function sendToNative(message) {
  const port = ensureNativePort();
  if (!port) {
    return false;
  }
  try {
    port.postMessage(message);
    return true;
  } catch (err) {
    setLastError('Native host недоступен: ' + err.message);
    return false;
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'startRecording') {
    (async () => {
      try {
        setLastError(null);
        await ensureOffscreenDocument();
        const res = await chrome.runtime.sendMessage({
          type: 'start',
          streamId: message.streamId,
          includeMic: message.includeMic,
        });
        if (!res || res.ok !== true) {
          const error = (res && res.error) ? res.error : 'не удалось запустить запись';
          setLastError(error);
          sendResponse({ ok: false, error });
          return;
        }
        sendResponse({ ok: true });
      } catch (e) {
        const error = e && e.message ? e.message : 'не удалось запустить запись';
        setLastError(error);
        sendResponse({ ok: false, error });
      }
    })();
    return true;
  }

  if (message.type === 'recording-started') {
    isRecording = true;
    return false;
  }

  if (message.type === 'stopRecording') {
    chrome.runtime.sendMessage({ type: 'stop' }).catch(() => {});
    sendResponse({ ok: true });
    return true;
  }

  if (message.type === 'getRecordingState') {
    sendResponse({ isRecording, lastError });
    return false;
  }

  if (message.type === 'stream_start') {
    const ok = sendToNative({
      type: 'stream_start',
      session_id: message.sessionId,
      mime_type: message.mimeType,
    });
    sendResponse({ ok });
    return true;
  }

  if (message.type === 'stream_chunk') {
    const ok = sendToNative({
      type: 'stream_chunk',
      session_id: message.sessionId,
      index: message.index,
      data: message.data,
    });
    sendResponse({ ok });
    return true;
  }

  if (message.type === 'stream_stop') {
    const ok = sendToNative({
      type: 'stream_stop',
      session_id: message.sessionId,
    });
    sendResponse({ ok });
    return true;
  }

  if (message.type === 'recording-stopped') {
    isRecording = false;
    closeOffscreenDocument();
    return false;
  }

  return false;
});
