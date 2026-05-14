// Frontend controller for the landing-page translator experience.
const video = document.getElementById('webcam');
const canvas = document.getElementById('captureCanvas');
const cameraStatus = document.getElementById('cameraStatus');
const liveDot = document.querySelector('.live-dot');
const detectedLetter = document.getElementById('detectedLetter');
const confidenceValue = document.getElementById('confidenceValue');
const formedWord = document.getElementById('formedWord');
const sentenceOutput = document.getElementById('sentenceOutput');
const gestureHistory = document.getElementById('gestureHistory');
const assistantInput = document.getElementById('assistantInput');
const assistantOutput = document.getElementById('assistantOutput');
const assistantSource = document.getElementById('assistantSource');
const speechText = document.getElementById('speechText');
const speechStatus = document.getElementById('speechStatus');
const ttsInput = document.getElementById('ttsInput');
const dashboardDetected = document.getElementById('dashboardDetected');
const dashboardConfidence = document.getElementById('dashboardConfidence');

let mediaStream = null;
let predictionTimer = null;
let chart = null;
let recognition = null;
let isListening = false;
let finalTranscript = '';
let speechRetryUsed = false;
const gestureCounts = { A: 0, E: 0, H: 0, L: 0, O: 0, R: 0, S: 0, T: 0 };

function initChart() {
  const ctx = document.getElementById('gestureChart');
  if (!ctx || typeof Chart === 'undefined') {
    console.warn('Chart.js is unavailable; analytics chart disabled.');
    return;
  }

  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Object.keys(gestureCounts),
      datasets: [{
        label: 'Most frequent gestures',
        data: Object.values(gestureCounts),
        backgroundColor: '#0b7285',
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
    }
  });
}

async function startCamera() {
  mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = mediaStream;
  cameraStatus.textContent = 'Camera running';
  liveDot.classList.add('active');
  predictionTimer = window.setInterval(sendFrameForPrediction, 2200);
}

function stopCamera() {
  if (predictionTimer) {
    window.clearInterval(predictionTimer);
    predictionTimer = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
  video.srcObject = null;
  cameraStatus.textContent = 'Camera idle';
  liveDot.classList.remove('active');
}

async function sendFrameForPrediction() {
  if (!mediaStream || video.readyState < 2) return;

  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const image = canvas.toDataURL('image/jpeg', 0.75);

  const response = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image,
      word: formedWord.value,
      sentence: sentenceOutput.value
    })
  });
  const data = await response.json();
  updatePrediction(data.letter, data.confidence);
}

function updatePrediction(letter, confidence) {
  detectedLetter.textContent = letter;
  confidenceValue.textContent = `${confidence}%`;
  dashboardDetected.textContent = letter;
  dashboardConfidence.textContent = `${confidence}%`;

  formedWord.value += letter.toLowerCase();
  sentenceOutput.value = formedWord.value;
  assistantInput.value = sentenceOutput.value;

  gestureCounts[letter] = (gestureCounts[letter] || 0) + 1;
  if (chart) {
    chart.data.datasets[0].data = Object.values(gestureCounts);
    chart.update();
  }

  const item = document.createElement('li');
  item.textContent = `${letter} ${confidence}%`;
  gestureHistory.prepend(item);
  while (gestureHistory.children.length > 8) {
    gestureHistory.removeChild(gestureHistory.lastChild);
  }
}

function clearText() {
  formedWord.value = '';
  sentenceOutput.value = '';
  assistantInput.value = '';
  detectedLetter.textContent = '-';
  confidenceValue.textContent = '0%';
  dashboardDetected.textContent = '-';
  dashboardConfidence.textContent = '0%';
  gestureHistory.innerHTML = '';
}

function speak(text) {
  if (!text.trim()) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.95;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

async function enhanceText(mode) {
  const text = assistantInput.value || sentenceOutput.value;
  if (!text.trim()) return;

  assistantOutput.value = 'Working...';
  const response = await fetch('/enhance-text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, mode })
  });
  const data = await response.json();
  assistantOutput.value = data.enhanced_text || text;
  assistantSource.textContent = data.message ? `Source: ${data.source}. ${data.message}` : `Source: ${data.source}`;
}

function getSpeechRecognition(useSimpleMode = false) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    return null;
  }

  const recognizer = new SpeechRecognition();
  // Some browsers return a "network" error with continuous mode. The retry path
  // below switches to simple mode automatically.
  recognizer.continuous = !useSimpleMode;
  recognizer.interimResults = true;
  recognizer.lang = 'en-US';

  recognizer.onstart = () => {
    isListening = true;
    speechStatus.textContent = 'Listening... speak now';
    document.getElementById('startListening').textContent = 'Stop Listening';
  };

  recognizer.onresult = event => {
    let interimTranscript = '';
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += `${transcript} `;
      } else {
        interimTranscript += transcript;
      }
    }
    speechText.value = `${finalTranscript}${interimTranscript}`.trim();
    ttsInput.value = speechText.value;
  };

  recognizer.onerror = event => {
    isListening = false;
    document.getElementById('startListening').textContent = 'Start Listening';
    if (event.error === 'not-allowed') {
      speechStatus.textContent = 'Microphone blocked. Allow mic permission in the browser.';
    } else if (event.error === 'no-speech') {
      speechStatus.textContent = 'No speech detected. Try again closer to the microphone.';
    } else if (event.error === 'network' && !speechRetryUsed) {
      speechRetryUsed = true;
      speechStatus.textContent = 'Speech service network issue. Retrying in simple mode...';
      window.setTimeout(() => {
        recognition = getSpeechRecognition(true);
        if (recognition) {
          try {
            recognition.start();
          } catch (error) {
            speechStatus.textContent = 'Speech retry failed. Try Chrome or Edge with microphone permission enabled.';
          }
        }
      }, 500);
    } else if (event.error === 'network') {
      speechStatus.textContent = 'Browser speech service network error. Try Chrome/Edge, disable Brave Shields for localhost, or check internet access.';
    } else {
      speechStatus.textContent = `Speech error: ${event.error}`;
    }
  };

  recognizer.onend = () => {
    isListening = false;
    document.getElementById('startListening').textContent = 'Start Listening';
    if (speechStatus.textContent === 'Listening... speak now') {
      speechStatus.textContent = 'Listening stopped';
    }
  };

  return recognizer;
}

function toggleListening() {
  if (isListening && recognition) {
    recognition.stop();
    return;
  }

  speechRetryUsed = false;
  recognition = getSpeechRecognition(false);
  if (!recognition) {
    speechStatus.textContent = 'Speech recognition is not supported here. Use Chrome or Edge.';
    speechText.value = 'Your browser does not expose the Web Speech Recognition API. Try Google Chrome or Microsoft Edge on http://127.0.0.1:5000/.';
    return;
  }

  try {
    recognition.start();
  } catch (error) {
    speechStatus.textContent = 'Speech recognition is already starting. Please wait a moment.';
  }
}

document.getElementById('startCamera').addEventListener('click', startCamera);
document.getElementById('stopCamera').addEventListener('click', stopCamera);
document.getElementById('clearText').addEventListener('click', clearText);
document.getElementById('speakOutput').addEventListener('click', () => speak(sentenceOutput.value));
document.getElementById('speakCustom').addEventListener('click', () => speak(ttsInput.value));
document.getElementById('improveText').addEventListener('click', () => enhanceText('improve'));
document.getElementById('generateEmail').addEventListener('click', () => enhanceText('email'));
document.getElementById('startListening').addEventListener('click', toggleListening);

initChart();
