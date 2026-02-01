let mediaRecorder = null;
let tabStream = null;
let micStream = null;
let audioContext = null;
let chunks = [];

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

function triggerDownload(blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `запись-встречи-${Date.now()}.webm`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function notifyStopped() {
  chrome.runtime.sendMessage({ type: 'recording-stopped' }).catch(() => {});
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
      .then(([tab, mic]) => {
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
        mediaRecorder = new MediaRecorder(destination.stream);
        chunks = [];

        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
          stopStreams();
          if (chunks.length > 0) {
            const blob = new Blob(chunks, { type: mimeType });
            triggerDownload(blob);
          }
          notifyStopped();
        };

        mediaRecorder.onerror = () => {
          stopStreams();
          notifyStopped();
        };

        mediaRecorder.start(1000);
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
