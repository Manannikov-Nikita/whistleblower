let mediaRecorder = null;
let tabStream = null;
let micStream = null;
let audioContext = null;
let chunks = [];
let currentSessionId = null;
let sendQueue = Promise.resolve();

const TIMESLICE_MS = 1000;
const MAX_CHUNK_BYTES = 256 * 1024;

function stopStreams() {
  if (tabStream) {
    tabStream.getTracks().forEach((t) => t.stop());
    tabStream = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((t) => t.stop());
    micStream = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
    audioContext.close();
    audioContext = null;
  }
}

function notifyStopped() {
  chrome.runtime.sendMessage({ type: 'recording-stopped' }).catch(() => {});
}

function arrayBufferToBase64(buffer) {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.length;
  for (let i = 0; i < len; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function sendNativeMessage(payload) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(payload, (res) => {
      resolve(res);
    });
  });
}

function sendChunk(sessionId, index, data) {
  return sendNativeMessage({
    type: 'stream_chunk',
    sessionId,
    index,
    data,
  });
}

function enqueueBlob(sessionId, blob) {
  const total = blob.size;
  let offset = 0;
  let index = 0;

  while (offset < total) {
    const slice = blob.slice(offset, offset + MAX_CHUNK_BYTES);
    offset += MAX_CHUNK_BYTES;
    const currentIndex = index;
    index += 1;

    sendQueue = sendQueue.then(async () => {
      const buffer = await slice.arrayBuffer();
      const data = arrayBufferToBase64(buffer);
      await sendChunk(sessionId, currentIndex, data);
    }).catch(() => {});
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === 'start') {
    const { streamId, includeMic } = message;

    const startTab = () => {
      return navigator.mediaDevices.getUserMedia({
        audio: {
          mandatory: {
            chromeMediaSource: 'tab',
            chromeMediaSourceId: streamId,
          },
        },
      });
    };

    const startMic = () => {
      if (!includeMic) return Promise.resolve(null);
      return navigator.mediaDevices.getUserMedia({ audio: true });
    };

    Promise.all([startTab(), startMic()])
      .then(async ([tab, mic]) => {
        tabStream = tab;
        micStream = mic;

        audioContext = new AudioContext();
        const destination = audioContext.createMediaStreamDestination();

        const tabSource = audioContext.createMediaStreamSource(tabStream);
        tabSource.connect(destination);
        tabSource.connect(audioContext.destination);

        if (micStream) {
          const micSource = audioContext.createMediaStreamSource(micStream);
          micSource.connect(destination);
        }

        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm';
        mediaRecorder = new MediaRecorder(destination.stream, { mimeType });
        chunks = [];

        currentSessionId = `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
        await sendNativeMessage({
          type: 'stream_start',
          sessionId: currentSessionId,
          mimeType,
        });

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0 && currentSessionId) {
            enqueueBlob(currentSessionId, e.data);
          }
        };

        mediaRecorder.onstop = () => {
          stopStreams();
          const sessionId = currentSessionId;
          currentSessionId = null;
          sendQueue = sendQueue.then(() => {
            if (!sessionId) return null;
            return sendNativeMessage({
              type: 'stream_stop',
              sessionId,
            });
          }).finally(() => {
            notifyStopped();
          });
        };

        mediaRecorder.onerror = () => {
          stopStreams();
          notifyStopped();
        };

        mediaRecorder.start(TIMESLICE_MS);
        chrome.runtime.sendMessage({ type: 'recording-started' }).catch(() => {});
        sendResponse({ ok: true });
      })
      .catch((err) => {
        stopStreams();
        notifyStopped();
        sendResponse({ ok: false, error: err.message });
      });

    return true;
  }

  if (message.type === 'stop') {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
    sendResponse({ ok: true });
    return true;
  }

  return false;
});
