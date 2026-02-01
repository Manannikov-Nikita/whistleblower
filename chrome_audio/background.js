const OFFSCREEN_URL = 'offscreen.html';
let isRecording = false;

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

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'startRecording') {
    (async () => {
      try {
        await ensureOffscreenDocument();
        chrome.runtime.sendMessage({
          type: 'start',
          streamId: message.streamId,
          includeMic: message.includeMic,
        }).catch(() => {});
        sendResponse({ ok: true });
      } catch (e) {
        sendResponse({ ok: false, error: e.message });
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
    sendResponse({ isRecording });
    return false;
  }

  if (message.type === 'recording-stopped') {
    isRecording = false;
    closeOffscreenDocument();
    return false;
  }

  return false;
});
