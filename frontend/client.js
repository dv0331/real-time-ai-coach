/**
 * Actor's AI Coach - Frontend Client
 * 
 * Real-time rehearsal assistant for actors
 * Handles camera/mic capture, WebSocket communication, and UI updates
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const CONFIG = {
    // WebSocket - Auto-detect protocol (ws:// for HTTP, wss:// for HTTPS)
    WS_URL: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
    RECONNECT_DELAY: 2000,
    
    // Audio capture
    AUDIO_SAMPLE_RATE: 16000,
    AUDIO_CHUNK_MS: 100,
    
    // Video capture
    VIDEO_FPS: 8,
    VIDEO_WIDTH: 640,
    VIDEO_HEIGHT: 480,
    JPEG_QUALITY: 0.7,
    
    // UI
    MAX_TIPS: 5,
    TIP_EXPIRE_MS: 8000,
    
    // Filler words to highlight (actors need to eliminate these!)
    FILLER_WORDS: ['um', 'uh', 'like', 'you know', 'so', 'actually', 
                   'basically', 'literally', 'right', 'okay', 'well', 'er', 'ah']
};

// Performance modes for actors
const PERFORMANCE_MODES = {
    free_practice: { icon: '🎬', name: 'Free Practice', desc: 'Open rehearsal' },
    monologue: { icon: '🎤', name: 'Monologue', desc: 'Solo piece practice' },
    self_tape: { icon: '📹', name: 'Self-Tape Prep', desc: 'Audition preparation' },
    cold_read: { icon: '📄', name: 'Cold Read', desc: 'First time with material' }
};

// Scene intensity levels
const INTENSITY_LEVELS = {
    intimate: { emoji: '🤫', name: 'Intimate' },
    neutral: { emoji: '😐', name: 'Neutral' },
    heightened: { emoji: '😢', name: 'Heightened' },
    intense: { emoji: '😤', name: 'Intense' }
};

// ============================================================================
// STATE
// ============================================================================

let state = {
    ws: null,
    isConnected: false,
    isSessionActive: false,
    
    // Media streams
    mediaStream: null,
    audioContext: null,
    audioProcessor: null,
    
    // Capture intervals
    videoInterval: null,
    
    // Session data
    sessionStartTime: null,
    timerInterval: null,
    
    // Transcript
    fullTranscript: '',
    wordCount: 0,
    
    // Tips
    activeTips: [],
    tipTimeouts: {},
    
    // Performance settings
    selectedMode: 'free_practice',
    sceneIntensity: 'neutral'
};

// ============================================================================
// DOM ELEMENTS
// ============================================================================

const elements = {
    // Video
    videoPreview: document.getElementById('videoPreview'),
    videoCanvas: document.getElementById('videoCanvas'),
    
    // Indicators
    faceIndicator: document.getElementById('faceIndicator'),
    gazeIndicator: document.getElementById('gazeIndicator'),
    speakingIndicator: document.getElementById('speakingIndicator'),
    
    // Controls
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    
    // Status
    statusBadge: document.getElementById('statusBadge'),
    sessionTimer: document.getElementById('sessionTimer'),
    
    // Mode selector
    modeSelector: document.getElementById('modeSelector'),
    modeBadge: document.getElementById('modeBadge'),
    modeDropdown: document.getElementById('modeDropdown'),
    
    // Metrics
    overallScore: document.getElementById('overallScore'),
    overallProgress: document.getElementById('overallProgress'),
    trendIndicator: document.getElementById('trendIndicator'),
    
    // Individual metrics
    paceBar: document.getElementById('paceBar'),
    paceValue: document.getElementById('paceValue'),
    paceDetail: document.getElementById('paceDetail'),
    pauseBar: document.getElementById('pauseBar'),
    pauseValue: document.getElementById('pauseValue'),
    energyBar: document.getElementById('energyBar'),
    energyValue: document.getElementById('energyValue'),
    energyDetail: document.getElementById('energyDetail'),
    varietyBar: document.getElementById('varietyBar'),
    varietyValue: document.getElementById('varietyValue'),
    eyeContactBar: document.getElementById('eyeContactBar'),
    eyeContactValue: document.getElementById('eyeContactValue'),
    presenceBar: document.getElementById('presenceBar'),
    presenceValue: document.getElementById('presenceValue'),
    fillersBar: document.getElementById('fillersBar'),
    fillersValue: document.getElementById('fillersValue'),
    fillersDetail: document.getElementById('fillersDetail'),
    stabilityBar: document.getElementById('stabilityBar'),
    stabilityValue: document.getElementById('stabilityValue'),
    
    // Coaching
    tipsList: document.getElementById('tipsList'),
    transcriptBox: document.getElementById('transcriptBox'),
    wordCount: document.getElementById('wordCount'),
    
    // Modal
    summaryModal: document.getElementById('summaryModal'),
    summaryContent: document.getElementById('summaryContent'),
    closeSummaryBtn: document.getElementById('closeSummaryBtn'),
    newSessionBtn: document.getElementById('newSessionBtn')
};

// ============================================================================
// WEBSOCKET
// ============================================================================

function connectWebSocket() {
    console.log('🎭 Connecting to Actor\'s Coach server:', CONFIG.WS_URL);
    
    state.ws = new WebSocket(CONFIG.WS_URL);
    
    state.ws.onopen = () => {
        console.log('✅ Connected to server');
        state.isConnected = true;
        updateStatus('connected', 'Ready');
    };
    
    state.ws.onclose = () => {
        console.log('❌ Disconnected from server');
        state.isConnected = false;
        updateStatus('disconnected', 'Disconnected');
        
        // Attempt reconnect
        setTimeout(connectWebSocket, CONFIG.RECONNECT_DELAY);
    };
    
    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    state.ws.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleMessage(message);
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };
}

function sendMessage(type, data = {}) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({
            type,
            ...data,
            mode: state.selectedMode,
            intensity: state.sceneIntensity,
            timestamp: Date.now() / 1000
        }));
    }
}

function handleMessage(message) {
    const { type, data } = message;
    
    switch (type) {
        case 'status':
            console.log('Status:', data.status, data.message);
            break;
            
        case 'metrics':
            updateMetrics(data);
            break;
            
        case 'transcript':
            updateTranscript(data);
            break;
            
        case 'tip':
            addTips(data.tips);
            break;
            
        case 'llm_tip':
            // Rich LLM coaching tip from Ollama
            handleLLMTip(data);
            break;
            
        case 'emotion':
            // Real-time emotion analysis
            updateEmotionDisplay(data);
            break;
            
        case 'summary':
            showSummary(data);
            break;
            
        default:
            console.log('Unknown message type:', type);
    }
}

// Handle structured LLM coaching tip
function handleLLMTip(data) {
    const tipElement = document.createElement('div');
    tipElement.className = 'tip tip-director llm-tip';
    tipElement.innerHTML = `
        <div class="llm-tip-header">
            <span class="tip-icon">🎬</span>
            <span class="llm-tip-label">Director's Note</span>
        </div>
        <div class="llm-tip-content">
            <div class="llm-tip-main">${data.tip || 'Keep going!'}</div>
            ${data.strength ? `<div class="llm-tip-strength"><span class="strength-label">✓ Strength:</span> ${data.strength}</div>` : ''}
            ${data.improve ? `<div class="llm-tip-improve"><span class="improve-label">→ Focus:</span> ${data.improve}</div>` : ''}
            ${data.emotion_note ? `<div class="llm-tip-emotion"><span class="emotion-label">🎭</span> ${data.emotion_note}</div>` : ''}
        </div>
    `;
    
    // Add to tips list at top
    const tipsList = elements.tipsList;
    tipsList.insertBefore(tipElement, tipsList.firstChild);
    
    // Remove excess tips
    while (tipsList.children.length > CONFIG.MAX_TIPS) {
        tipsList.removeChild(tipsList.lastChild);
    }
    
    // Auto-remove after timeout
    setTimeout(() => {
        if (tipElement.parentNode) {
            tipElement.classList.add('fade-out');
            setTimeout(() => tipElement.remove(), 300);
        }
    }, CONFIG.TIP_EXPIRE_MS * 1.5);  // Director notes stay longer
}

// Update emotion display in UI
function updateEmotionDisplay(data) {
    // Audio emotion
    const audioEmotionEl = document.getElementById('audioEmotion');
    const arousalBar = document.getElementById('arousalBar');
    
    if (audioEmotionEl && data.audio) {
        const emotion = data.audio.emotion || 'neutral';
        const arousal = data.audio.arousal || 0.5;
        audioEmotionEl.textContent = `${getEmotionEmoji(emotion)} ${emotion}`;
        audioEmotionEl.title = `Arousal: ${(arousal * 100).toFixed(0)}%`;
        audioEmotionEl.dataset.emotion = emotion.toLowerCase();
        
        if (arousalBar) {
            arousalBar.style.width = `${arousal * 100}%`;
        }
    }
    
    // Face emotion
    const faceEmotionEl = document.getElementById('faceEmotion');
    const expressivenessBar = document.getElementById('expressivenessBar');
    
    if (faceEmotionEl && data.face) {
        const emotion = data.face.emotion || 'neutral';
        const expressiveness = data.face.expressiveness || 0.5;
        faceEmotionEl.textContent = `${getEmotionEmoji(emotion)} ${emotion}`;
        faceEmotionEl.title = `Expressiveness: ${(expressiveness * 100).toFixed(0)}%`;
        faceEmotionEl.dataset.emotion = emotion.toLowerCase();
        
        if (expressivenessBar) {
            expressivenessBar.style.width = `${expressiveness * 100}%`;
        }
    }
    
    // Text sentiment
    const textSentimentEl = document.getElementById('textSentiment');
    const textEmotionsListEl = document.getElementById('textEmotionsList');
    
    if (textSentimentEl && data.text) {
        const sentiment = data.text.sentiment || 'neutral';
        textSentimentEl.textContent = `${getSentimentEmoji(sentiment)} ${sentiment}`;
        
        // Update emotion tags
        if (textEmotionsListEl && data.text.emotions) {
            textEmotionsListEl.innerHTML = data.text.emotions
                .slice(0, 4)
                .map(e => `<span class="text-emotion-tag">${e.label} (${(e.score * 100).toFixed(0)}%)</span>`)
                .join('');
        }
    }
    
    // Overall intensity
    const emotionIntensityEl = document.getElementById('emotionIntensity');
    const intensityLabelEl = document.getElementById('intensityLabel');
    
    if (emotionIntensityEl && data.combined) {
        const intensity = data.combined.intensity || 0.5;
        emotionIntensityEl.style.width = `${intensity * 100}%`;
        emotionIntensityEl.className = `emotion-intensity-bar ${getIntensityClass(intensity)}`;
        
        if (intensityLabelEl) {
            intensityLabelEl.textContent = getIntensityLabel(intensity);
        }
    }
    
    // Update overall emotion state
    state.currentEmotion = data;
    
    console.log('🎭 Emotion update:', data);
}

// Get sentiment emoji
function getSentimentEmoji(sentiment) {
    const emojis = {
        'positive': '😊',
        'negative': '😔',
        'neutral': '😐'
    };
    return emojis[sentiment.toLowerCase()] || '😐';
}

// Get intensity label
function getIntensityLabel(intensity) {
    if (intensity > 0.75) return 'High';
    if (intensity > 0.5) return 'Moderate';
    if (intensity > 0.25) return 'Low';
    return 'Minimal';
}

// Get emoji for emotion
function getEmotionEmoji(emotion) {
    const emojis = {
        'neutral': '😐',
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fear': '😨',
        'disgust': '🤢',
        'surprise': '😲',
        'excited': '🤩',
        'calm': '😌',
        'joy': '😄',
        'love': '🥰',
        'anxious': '😰',
        'frustrated': '😤'
    };
    return emojis[emotion.toLowerCase()] || '🎭';
}

// Get intensity class for styling
function getIntensityClass(intensity) {
    if (intensity > 0.75) return 'intensity-high';
    if (intensity > 0.5) return 'intensity-medium';
    if (intensity > 0.25) return 'intensity-low';
    return 'intensity-minimal';
}

// ============================================================================
// MEDIA CAPTURE
// ============================================================================

async function initializeMedia() {
    try {
        // Request camera and microphone
        state.mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: CONFIG.VIDEO_WIDTH },
                height: { ideal: CONFIG.VIDEO_HEIGHT },
                facingMode: 'user'
            },
            audio: {
                sampleRate: { ideal: CONFIG.AUDIO_SAMPLE_RATE },
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
        
        // Show video preview
        elements.videoPreview.srcObject = state.mediaStream;
        
        // Initialize audio processing
        initializeAudioCapture();
        
        console.log('🎥 Camera and microphone ready');
        return true;
        
    } catch (error) {
        console.error('Failed to initialize media:', error);
        alert('Could not access camera/microphone. Please check permissions and try again.');
        return false;
    }
}

function initializeAudioCapture() {
    const audioTrack = state.mediaStream.getAudioTracks()[0];
    if (!audioTrack) return;
    
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: CONFIG.AUDIO_SAMPLE_RATE
    });
    
    const source = state.audioContext.createMediaStreamSource(state.mediaStream);
    
    // Create ScriptProcessor for audio capture
    // Buffer size must be a power of 2 between 256 and 16384
    // Use 2048 which provides ~128ms of audio at 16kHz - good balance of latency and stability
    const bufferSize = 2048;
    state.audioProcessor = state.audioContext.createScriptProcessor(bufferSize, 1, 1);
    
    state.audioProcessor.onaudioprocess = (event) => {
        if (!state.isSessionActive) return;
        
        const inputData = event.inputBuffer.getChannelData(0);
        
        // Convert float32 to int16
        const int16Data = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        // Send to server
        const base64 = arrayBufferToBase64(int16Data.buffer);
        sendMessage('audio', { data: base64 });
    };
    
    source.connect(state.audioProcessor);
    state.audioProcessor.connect(state.audioContext.destination);
}

function startVideoCapture() {
    const canvas = elements.videoCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = CONFIG.VIDEO_WIDTH;
    canvas.height = CONFIG.VIDEO_HEIGHT;
    
    state.videoInterval = setInterval(() => {
        if (!state.isSessionActive) return;
        
        // Draw current video frame to canvas
        ctx.drawImage(elements.videoPreview, 0, 0, canvas.width, canvas.height);
        
        // Get JPEG data
        canvas.toBlob((blob) => {
            if (blob) {
                blob.arrayBuffer().then((buffer) => {
                    const base64 = arrayBufferToBase64(buffer);
                    sendMessage('video', { data: base64 });
                });
            }
        }, 'image/jpeg', CONFIG.JPEG_QUALITY);
        
    }, 1000 / CONFIG.VIDEO_FPS);
}

function stopVideoCapture() {
    if (state.videoInterval) {
        clearInterval(state.videoInterval);
        state.videoInterval = null;
    }
}

// ============================================================================
// SESSION MANAGEMENT
// ============================================================================

async function startSession() {
    if (!state.isConnected) {
        alert('Not connected to server. Please wait for connection...');
        return;
    }
    
    // Initialize media if not already done
    if (!state.mediaStream) {
        const success = await initializeMedia();
        if (!success) return;
    }
    
    // Resume audio context if suspended
    if (state.audioContext && state.audioContext.state === 'suspended') {
        await state.audioContext.resume();
    }
    
    // Start session
    state.isSessionActive = true;
    state.sessionStartTime = Date.now();
    state.fullTranscript = '';
    state.wordCount = 0;
    
    // Start session recording (for history)
    if (typeof startSessionRecording === 'function') {
        startSessionRecording();
    }
    
    // Start capturing
    startVideoCapture();
    
    // Start timer
    state.timerInterval = setInterval(updateTimer, 1000);
    
    // Update UI
    elements.startBtn.classList.add('hidden');
    elements.stopBtn.classList.remove('hidden');
    updateStatus('recording', 'Recording');
    
    // Clear tips with actor-focused message
    elements.tipsList.innerHTML = `
        <div class="tip tip-success">
            <span class="tip-icon">🎬</span>
            <span class="tip-text">Action! Begin your performance...</span>
        </div>
    `;
    
    // Clear transcript
    elements.transcriptBox.innerHTML = '<p class="transcript-text"></p>';
    elements.wordCount.textContent = '0 words';
    
    // Send start command
    sendMessage('control', { action: 'start' });
    
    console.log('🎬 Session started');
}

function stopSession() {
    state.isSessionActive = false;
    
    // Stop session recording (for history) - do this BEFORE stopping everything else
    if (typeof stopSessionRecording === 'function') {
        stopSessionRecording();
    }
    
    // Stop capturing
    stopVideoCapture();
    
    // Stop timer
    if (state.timerInterval) {
        clearInterval(state.timerInterval);
        state.timerInterval = null;
    }
    
    // Update UI
    elements.startBtn.classList.remove('hidden');
    elements.stopBtn.classList.add('hidden');
    updateStatus('connected', 'Session Complete');
    
    // Send stop command
    sendMessage('control', { action: 'stop' });
    
    console.log('🎬 Cut! Session ended');
}

function updateTimer() {
    if (!state.sessionStartTime) return;
    
    const elapsed = Math.floor((Date.now() - state.sessionStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');
    
    elements.sessionTimer.textContent = `${minutes}:${seconds}`;
}

// ============================================================================
// UI UPDATES
// ============================================================================

function updateStatus(status, text) {
    const badge = elements.statusBadge;
    badge.className = 'status-badge';
    
    if (status === 'recording') {
        badge.classList.add('recording');
    } else if (status === 'connected') {
        badge.classList.add('active');
    }
    
    badge.querySelector('.status-text').textContent = text;
}

function updateMetrics(data) {
    const { scores, raw, flags, trend } = data;
    
    // Update overall score
    const overallPercent = Math.round(scores.overall * 100);
    elements.overallScore.textContent = overallPercent;
    
    // Update ring progress (circumference = 2 * PI * 45 = 283)
    const offset = 283 - (283 * scores.overall);
    elements.overallProgress.style.strokeDashoffset = offset;
    
    // Update trend indicator
    elements.trendIndicator.className = 'trend-indicator';
    if (trend === 1) {
        elements.trendIndicator.classList.add('improving');
        elements.trendIndicator.querySelector('.trend-arrow').textContent = '↗';
        elements.trendIndicator.querySelector('.trend-text').textContent = 'Improving';
    } else if (trend === -1) {
        elements.trendIndicator.classList.add('declining');
        elements.trendIndicator.querySelector('.trend-arrow').textContent = '↘';
        elements.trendIndicator.querySelector('.trend-text').textContent = 'Declining';
    } else {
        elements.trendIndicator.querySelector('.trend-arrow').textContent = '→';
        elements.trendIndicator.querySelector('.trend-text').textContent = 'Stable';
    }
    
    // Update individual metrics
    updateMetricBar('pace', scores.pace, `${Math.round(raw.wpm)} WPM`);
    updateMetricBar('pause', scores.stability || 0.5);  // Using stability as pause proxy
    updateMetricBar('energy', scores.energy, `${raw.energy_db.toFixed(0)} dB`);
    updateMetricBar('variety', scores.pitch_variety);
    updateMetricBar('eyeContact', scores.eye_contact);
    updateMetricBar('presence', scores.presence);
    updateMetricBar('fillers', scores.filler_words, `${(raw.filler_ratio * 100).toFixed(1)}%`);
    updateMetricBar('stability', scores.stability);
    
    // Update face/gaze indicators
    if (!flags.face_detected) {
        elements.faceIndicator.classList.add('visible', 'warning');
        elements.faceIndicator.querySelector('.indicator-text').textContent = 'Find your light!';
    } else {
        elements.faceIndicator.classList.remove('visible', 'warning');
    }
    
    if (flags.face_detected && !flags.looking_at_camera) {
        elements.gazeIndicator.classList.add('visible', 'warning');
        elements.gazeIndicator.querySelector('.indicator-text').textContent = 'Eye contact!';
    } else {
        elements.gazeIndicator.classList.remove('visible', 'warning');
    }
    
    // Update speaking indicator
    if (elements.speakingIndicator) {
        if (flags.is_speaking) {
            elements.speakingIndicator.classList.add('active');
        } else {
            elements.speakingIndicator.classList.remove('active');
        }
    }
}

function updateMetricBar(metric, score, detail = null) {
    const bar = elements[`${metric}Bar`];
    const value = elements[`${metric}Value`];
    const detailEl = elements[`${metric}Detail`];
    
    if (!bar || !value) return;
    
    const percent = Math.round(score * 100);
    bar.style.width = `${percent}%`;
    value.textContent = percent;
    
    // Color based on score
    bar.classList.remove('low', 'medium', 'high');
    if (score < 0.4) {
        bar.classList.add('low');
    } else if (score < 0.7) {
        bar.classList.add('medium');
    } else {
        bar.classList.add('high');
    }
    
    if (detail && detailEl) {
        detailEl.textContent = detail;
    }
}

function updateTranscript(data) {
    const { text, words, is_partial } = data;
    
    if (!text) return;
    
    // Append to full transcript
    state.fullTranscript += ' ' + text;
    state.fullTranscript = state.fullTranscript.trim();
    
    // Update word count
    state.wordCount = state.fullTranscript.split(/\s+/).filter(w => w).length;
    elements.wordCount.textContent = `${state.wordCount} words`;
    
    // Format with filler word highlighting
    const formattedText = formatTranscript(state.fullTranscript);
    
    elements.transcriptBox.innerHTML = `<p class="transcript-text">${formattedText}</p>`;
    
    // Scroll to bottom
    elements.transcriptBox.scrollTop = elements.transcriptBox.scrollHeight;
}

function formatTranscript(text) {
    let formatted = text;
    
    // Highlight filler words (important for actors!)
    CONFIG.FILLER_WORDS.forEach(filler => {
        const regex = new RegExp(`\\b(${filler})\\b`, 'gi');
        formatted = formatted.replace(regex, '<span class="filler">$1</span>');
    });
    
    return formatted;
}

function addTips(tips) {
    tips.forEach(tip => addTip(tip));
}

function addTip(tip) {
    // Create tip element
    const tipEl = document.createElement('div');
    tipEl.className = `tip tip-${tip.severity}`;
    tipEl.dataset.tipId = tip.id;
    
    // Choose icon based on category (actor-focused)
    const icons = {
        'delivery': '🗣️',
        'vocal': '🎵',
        'presence': '✨',
        'technique': '🎯',
        'general': '💡',
        'director': '🎬',
        'pace': '⏱️',
        'energy': '⚡',
        'eye_contact': '👁️',
        'fillers': '💬'
    };
    
    tipEl.innerHTML = `
        <span class="tip-icon">${icons[tip.category] || '💡'}</span>
        <span class="tip-text">${tip.message}</span>
    `;
    
    // Remove old tip of same type
    const existing = elements.tipsList.querySelector(`[data-tip-id="${tip.id}"]`);
    if (existing) {
        existing.remove();
    }
    
    // Add new tip at top
    elements.tipsList.insertBefore(tipEl, elements.tipsList.firstChild);
    
    // Limit number of tips
    while (elements.tipsList.children.length > CONFIG.MAX_TIPS) {
        elements.tipsList.lastChild.remove();
    }
    
    // Auto-remove after timeout
    if (state.tipTimeouts[tip.id]) {
        clearTimeout(state.tipTimeouts[tip.id]);
    }
    state.tipTimeouts[tip.id] = setTimeout(() => {
        tipEl.style.opacity = '0';
        setTimeout(() => tipEl.remove(), 300);
    }, CONFIG.TIP_EXPIRE_MS);
}

function showSummary(data) {
    const { session_id, duration_seconds, final_scores, audio_stats, coach_stats, transcript } = data;
    
    const minutes = Math.floor(duration_seconds / 60);
    const seconds = Math.round(duration_seconds % 60);
    
    // Generate actor-focused summary
    elements.summaryContent.innerHTML = `
        <div class="summary-section">
            <h3>📊 Session Overview</h3>
            <div class="summary-stats">
                <div class="summary-stat">
                    <div class="summary-stat-value">${minutes}:${seconds.toString().padStart(2, '0')}</div>
                    <div class="summary-stat-label">Duration</div>
                </div>
                <div class="summary-stat">
                    <div class="summary-stat-value">${audio_stats?.total_words || state.wordCount}</div>
                    <div class="summary-stat-label">Words</div>
                </div>
                <div class="summary-stat">
                    <div class="summary-stat-value">${Math.round(audio_stats?.average_wpm || 0)}</div>
                    <div class="summary-stat-label">Avg WPM</div>
                </div>
                <div class="summary-stat">
                    <div class="summary-stat-value">${audio_stats?.filler_words || 0}</div>
                    <div class="summary-stat-label">Fillers</div>
                </div>
            </div>
        </div>
        
        <div class="summary-section">
            <h3>🎭 Performance Scores</h3>
            <div class="summary-scores">
                ${Object.entries(final_scores || {}).map(([key, value]) => `
                    <span class="summary-score-badge">
                        <span class="label">${formatScoreLabel(key)}:</span>
                        <span class="value">${Math.round(value * 100)}</span>
                    </span>
                `).join('')}
            </div>
        </div>
        
        <div class="summary-section">
            <h3>🎬 Director's Notes</h3>
            <div class="summary-stats">
                <div class="summary-stat">
                    <div class="summary-stat-value">${coach_stats?.total_tips_given || 0}</div>
                    <div class="summary-stat-label">Notes Given</div>
                </div>
                <div class="summary-stat">
                    <div class="summary-stat-value">${coach_stats?.positive_tips || 0}</div>
                    <div class="summary-stat-label">Positive</div>
                </div>
            </div>
            ${coach_stats?.areas_to_work_on?.length > 0 ? `
                <p style="margin-top: 16px; color: var(--text-secondary);">
                    <strong>Focus areas:</strong> ${coach_stats.areas_to_work_on.join(', ')}
                </p>
            ` : ''}
        </div>
    `;
    
    elements.summaryModal.classList.remove('hidden');
}

function formatScoreLabel(key) {
    const labels = {
        'overall': 'Overall',
        'pace': 'Pace',
        'energy': 'Energy',
        'pitch_variety': 'Vocal Variety',
        'filler_words': 'Technique',
        'eye_contact': 'Eye Contact',
        'presence': 'Presence',
        'stability': 'Stability'
    };
    return labels[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// ============================================================================
// UTILITIES
// ============================================================================

function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

// ============================================================================
// MODE & INTENSITY SELECTOR
// ============================================================================

function initModeSelector() {
    // Toggle dropdown
    elements.modeBadge.addEventListener('click', (e) => {
        e.stopPropagation();
        elements.modeSelector.classList.toggle('open');
        elements.modeDropdown.classList.toggle('hidden');
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', () => {
        elements.modeSelector.classList.remove('open');
        elements.modeDropdown.classList.add('hidden');
    });
    
    // Mode selection
    document.querySelectorAll('.mode-option').forEach(option => {
        option.addEventListener('click', (e) => {
            e.stopPropagation();
            const mode = option.dataset.mode;
            selectMode(mode);
            
            // Close dropdown
            elements.modeSelector.classList.remove('open');
            elements.modeDropdown.classList.add('hidden');
        });
    });
    
    // Intensity selection
    document.querySelectorAll('.intensity-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const intensity = btn.dataset.intensity;
            selectIntensity(intensity);
        });
    });
}

function selectMode(mode) {
    state.selectedMode = mode;
    const modeInfo = PERFORMANCE_MODES[mode];
    
    if (modeInfo) {
        // Update badge
        elements.modeBadge.querySelector('.mode-icon').textContent = modeInfo.icon;
        elements.modeBadge.querySelector('.mode-name').textContent = modeInfo.name;
        
        // Update active state in dropdown
        document.querySelectorAll('.mode-option').forEach(opt => {
            opt.classList.toggle('active', opt.dataset.mode === mode);
        });
        
        // Update tips
        elements.tipsList.innerHTML = `
            <div class="tip tip-info">
                <span class="tip-icon">${modeInfo.icon}</span>
                <span class="tip-text">${modeInfo.name} mode - ${modeInfo.desc}</span>
            </div>
        `;
        
        console.log(`🎭 Mode: ${modeInfo.name}`);
    }
}

function selectIntensity(intensity) {
    state.sceneIntensity = intensity;
    
    // Update active button
    document.querySelectorAll('.intensity-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.intensity === intensity);
    });
    
    const intensityInfo = INTENSITY_LEVELS[intensity];
    console.log(`🎭 Intensity: ${intensityInfo.name}`);
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

elements.startBtn.addEventListener('click', startSession);
elements.stopBtn.addEventListener('click', stopSession);

elements.closeSummaryBtn.addEventListener('click', () => {
    elements.summaryModal.classList.add('hidden');
});

elements.newSessionBtn?.addEventListener('click', () => {
    elements.summaryModal.classList.add('hidden');
    // Reset for new session
    state.fullTranscript = '';
    state.wordCount = 0;
    elements.transcriptBox.innerHTML = '<p class="transcript-placeholder">Your performance will be transcribed here...</p>';
    elements.wordCount.textContent = '0 words';
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Escape to stop session
    if (e.key === 'Escape' && state.isSessionActive) {
        stopSession();
    }
    // Space to start (when not in input)
    if (e.key === ' ' && !state.isSessionActive && e.target === document.body) {
        e.preventDefault();
        startSession();
    }
});

// ============================================================================
// INITIALIZATION
// ============================================================================

// Add SVG gradient for score ring
const svg = document.querySelector('.score-ring svg');
if (svg) {
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
        <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#d4af37"/>
            <stop offset="100%" style="stop-color:#f4d03f"/>
        </linearGradient>
    `;
    svg.insertBefore(defs, svg.firstChild);
}

// Initialize mode selector
initModeSelector();

// Connect to WebSocket on load
connectWebSocket();

// ============================================================================
// ALWAYS-ON CAMERA PREVIEW (Director's Monitor)
// ============================================================================

/**
 * Director's Philosophy: An actor must ALWAYS see themselves before performing.
 * Just like a mirror in a rehearsal room, the camera preview should be
 * available the moment they open the app.
 */

let cameraPreviewState = {
    initialized: false,
    stream: null,
    framingGuideVisible: true,
    overlayMode: 'full' // 'full', 'minimal', 'off'
};

async function initCameraPreview() {
    if (cameraPreviewState.initialized) return;
    
    try {
        console.log('🎥 Initializing Director\'s Monitor...');
        
        // Request camera access immediately
        cameraPreviewState.stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: CONFIG.VIDEO_WIDTH },
                height: { ideal: CONFIG.VIDEO_HEIGHT },
                facingMode: 'user'
            },
            audio: false // Audio only needed for recording
        });
        
        // Connect to video preview
        const videoPreview = elements.videoPreview;
        if (videoPreview) {
            videoPreview.srcObject = cameraPreviewState.stream;
            videoPreview.play().catch(e => console.log('Autoplay blocked:', e));
        }
        
        cameraPreviewState.initialized = true;
        
        // Add framing guide overlay
        createFramingGuide();
        
        // Add visual feedback overlay
        createVisualFeedbackOverlay();
        
        // Show success indicator
        showCameraReady();
        
        console.log('✅ Director\'s Monitor ready - You can see yourself!');
        
    } catch (error) {
        console.error('Camera preview error:', error);
        showCameraPrompt();
    }
}

function showCameraReady() {
    const container = document.querySelector('.video-container');
    if (!container) return;
    
    // Remove any existing prompt
    const existingPrompt = container.querySelector('.camera-prompt');
    if (existingPrompt) existingPrompt.remove();
    
    // Flash a brief "ready" indicator
    const ready = document.createElement('div');
    ready.className = 'camera-ready-flash';
    ready.innerHTML = '🎬 Camera Ready';
    container.appendChild(ready);
    
    setTimeout(() => ready.remove(), 2000);
}

function showCameraPrompt() {
    const container = document.querySelector('.video-container');
    if (!container) return;
    
    const prompt = document.createElement('div');
    prompt.className = 'camera-prompt';
    prompt.innerHTML = `
        <div class="prompt-content">
            <span class="prompt-icon">📷</span>
            <h3>Enable Your Camera</h3>
            <p>A director needs to see you! Click below to enable your camera preview.</p>
            <button class="btn btn-primary" id="enableCameraBtn">
                <span class="btn-icon">🎥</span>
                Enable Camera
            </button>
        </div>
    `;
    container.appendChild(prompt);
    
    document.getElementById('enableCameraBtn')?.addEventListener('click', () => {
        prompt.remove();
        initCameraPreview();
    });
}

// ============================================================================
// DIRECTOR'S FRAMING GUIDES (Professional Film Standards)
// ============================================================================

/**
 * Professional framing guides used in film and television:
 * - Rule of Thirds: Classic composition grid
 * - Eye Line: Where your eyes should be for close-ups
 * - Safe Zone: Broadcast safe area
 * - Head Room: Proper spacing above head
 */

function createFramingGuide() {
    const container = document.querySelector('.video-container');
    if (!container || container.querySelector('.framing-overlay')) return;
    
    const overlay = document.createElement('div');
    overlay.className = 'framing-overlay';
    overlay.innerHTML = `
        <!-- Rule of Thirds Grid -->
        <svg class="thirds-grid" viewBox="0 0 100 100" preserveAspectRatio="none">
            <!-- Vertical lines -->
            <line x1="33.33" y1="0" x2="33.33" y2="100" class="grid-line"/>
            <line x1="66.67" y1="0" x2="66.67" y2="100" class="grid-line"/>
            <!-- Horizontal lines -->
            <line x1="0" y1="33.33" x2="100" y2="33.33" class="grid-line"/>
            <line x1="0" y1="66.67" x2="100" y2="66.67" class="grid-line"/>
            <!-- Power points -->
            <circle cx="33.33" cy="33.33" r="1.5" class="power-point"/>
            <circle cx="66.67" cy="33.33" r="1.5" class="power-point"/>
            <circle cx="33.33" cy="66.67" r="1.5" class="power-point"/>
            <circle cx="66.67" cy="66.67" r="1.5" class="power-point"/>
        </svg>
        
        <!-- Eye Line Guide (for close-ups) -->
        <div class="eye-line-guide">
            <span class="eye-line-label">👁 Eye Line</span>
        </div>
        
        <!-- Headroom Guide -->
        <div class="headroom-guide">
            <span class="headroom-label">⬆ Headroom</span>
        </div>
        
        <!-- Center Mark -->
        <div class="center-mark">
            <div class="center-crosshair"></div>
        </div>
        
        <!-- Frame Controls -->
        <div class="frame-controls">
            <button class="frame-btn" id="toggleGridBtn" title="Toggle Grid">▦</button>
            <button class="frame-btn" id="toggleGuidesBtn" title="Toggle Guides">◎</button>
            <button class="frame-btn active" id="mirrorBtn" title="Mirror View">↔</button>
        </div>
    `;
    
    container.appendChild(overlay);
    
    // Add control listeners
    document.getElementById('toggleGridBtn')?.addEventListener('click', toggleGrid);
    document.getElementById('toggleGuidesBtn')?.addEventListener('click', toggleGuides);
    document.getElementById('mirrorBtn')?.addEventListener('click', toggleMirror);
}

function toggleGrid() {
    const grid = document.querySelector('.thirds-grid');
    const btn = document.getElementById('toggleGridBtn');
    if (grid) {
        grid.classList.toggle('hidden');
        btn?.classList.toggle('active');
    }
}

function toggleGuides() {
    const guides = document.querySelectorAll('.eye-line-guide, .headroom-guide, .center-mark');
    const btn = document.getElementById('toggleGuidesBtn');
    guides.forEach(g => g.classList.toggle('hidden'));
    btn?.classList.toggle('active');
}

function toggleMirror() {
    const video = elements.videoPreview;
    const btn = document.getElementById('mirrorBtn');
    if (video) {
        video.classList.toggle('no-mirror');
        btn?.classList.toggle('active');
    }
}

// ============================================================================
// REAL-TIME VISUAL FEEDBACK OVERLAY
// ============================================================================

/**
 * Show what the AI is seeing in real-time:
 * - Face detection box
 * - Emotion indicator
 * - Gaze direction
 * - Posture feedback
 */

function createVisualFeedbackOverlay() {
    const container = document.querySelector('.video-container');
    if (!container || container.querySelector('.ai-feedback-overlay')) return;
    
    const overlay = document.createElement('div');
    overlay.className = 'ai-feedback-overlay';
    overlay.innerHTML = `
        <!-- Face Detection Box -->
        <div class="face-box" id="faceBox">
            <div class="face-box-corner tl"></div>
            <div class="face-box-corner tr"></div>
            <div class="face-box-corner bl"></div>
            <div class="face-box-corner br"></div>
        </div>
        
        <!-- Live Emotion Badge -->
        <div class="live-emotion-badge" id="liveEmotionBadge">
            <span class="emotion-emoji">😐</span>
            <span class="emotion-text">Ready</span>
        </div>
        
        <!-- Gaze Indicator Arrow -->
        <div class="gaze-arrow" id="gazeArrow">
            <svg viewBox="0 0 24 24">
                <path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z"/>
            </svg>
        </div>
        
        <!-- Posture Alert -->
        <div class="posture-alert hidden" id="postureAlert">
            <span class="alert-icon">📐</span>
            <span class="alert-text">Adjust posture</span>
        </div>
        
        <!-- Performance Pulse (during recording) -->
        <div class="performance-pulse hidden" id="performancePulse">
            <div class="pulse-ring"></div>
            <span class="pulse-score">--</span>
        </div>
    `;
    
    container.appendChild(overlay);
}

// Update visual feedback based on metrics
function updateVisualFeedback(data) {
    const faceBox = document.getElementById('faceBox');
    const emotionBadge = document.getElementById('liveEmotionBadge');
    const gazeArrow = document.getElementById('gazeArrow');
    const postureAlert = document.getElementById('postureAlert');
    const performancePulse = document.getElementById('performancePulse');
    
    if (!data) return;
    
    // Update face box visibility and position
    if (faceBox) {
        if (data.flags?.face_detected) {
            faceBox.classList.add('detected');
            faceBox.classList.remove('lost');
            
            // If we have face coordinates, position the box
            if (data.face_bbox) {
                const { x, y, w, h } = data.face_bbox;
                faceBox.style.left = `${x * 100}%`;
                faceBox.style.top = `${y * 100}%`;
                faceBox.style.width = `${w * 100}%`;
                faceBox.style.height = `${h * 100}%`;
            }
        } else {
            faceBox.classList.remove('detected');
            faceBox.classList.add('lost');
        }
    }
    
    // Update live emotion badge
    if (emotionBadge && state.currentEmotion) {
        const faceEmotion = state.currentEmotion.face?.emotion || 'neutral';
        emotionBadge.querySelector('.emotion-emoji').textContent = getEmotionEmoji(faceEmotion);
        emotionBadge.querySelector('.emotion-text').textContent = faceEmotion;
        emotionBadge.dataset.emotion = faceEmotion.toLowerCase();
    }
    
    // Update gaze direction
    if (gazeArrow) {
        if (data.flags?.looking_at_camera) {
            gazeArrow.classList.add('centered');
            gazeArrow.classList.remove('looking-away');
        } else {
            gazeArrow.classList.remove('centered');
            gazeArrow.classList.add('looking-away');
        }
    }
    
    // Show performance pulse during session
    if (performancePulse && state.isSessionActive) {
        performancePulse.classList.remove('hidden');
        performancePulse.querySelector('.pulse-score').textContent = 
            Math.round((data.scores?.overall || 0.5) * 100);
    } else if (performancePulse) {
        performancePulse.classList.add('hidden');
    }
}

// Connect visual feedback to metrics updates
const originalUpdateMetrics = updateMetrics;
updateMetrics = function(data) {
    originalUpdateMetrics(data);
    updateVisualFeedback(data);
};

// ============================================================================
// TAB NAVIGATION
// ============================================================================

const tabs = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// Global switchTab function (can be called from HTML onclick)
function switchTab(targetTab) {
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Update active tab button
    tabs.forEach(t => {
        t.classList.remove('active');
        if (t.dataset.tab === targetTab) {
            t.classList.add('active');
        }
    });
    
    // Show target content
    tabContents.forEach(content => {
        content.classList.remove('active');
        if (content.id === `tab-${targetTab}`) {
            content.classList.add('active');
        }
    });
    
    // Initialize self-tape preview if switching to that tab
    if (targetTab === 'selftape') {
        initSelftapePreview();
    }
    
    // Load session history if switching to history tab
    if (targetTab === 'history') {
        if (typeof loadSessionHistory === 'function') {
            loadSessionHistory();
        }
    }
    
    console.log(`📑 Switched to tab: ${targetTab}`);
}

// Make switchTab available globally
window.switchTab = switchTab;

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        switchTab(tab.dataset.tab);
    });
});

// ============================================================================
// SELF-TAPE STUDIO
// ============================================================================

let selftapeState = {
    mediaStream: null,
    mediaRecorder: null,
    recordedChunks: [],
    isRecording: false,
    recordingStartTime: null,
    recordingTimer: null,
    takes: []
};

async function initSelftapePreview() {
    const preview = document.getElementById('selftapePreview');
    
    if (!preview) return;
    
    // If already have stream, just connect
    if (selftapeState.mediaStream) {
        preview.srcObject = selftapeState.mediaStream;
        return;
    }
    
    try {
        selftapeState.mediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1920 },
                height: { ideal: 1080 },
                facingMode: 'user'
            },
            audio: true
        });
        
        preview.srcObject = selftapeState.mediaStream;
        console.log('📹 Self-tape preview ready');
    } catch (error) {
        console.error('Failed to access camera:', error);
        alert('Could not access camera. Please check permissions.');
    }
}

function startSelftapeRecording() {
    if (!selftapeState.mediaStream) {
        alert('Camera not ready. Please wait...');
        return;
    }
    
    // Clear previous recording
    selftapeState.recordedChunks = [];
    
    // Create MediaRecorder
    const options = { mimeType: 'video/webm;codecs=vp9,opus' };
    try {
        selftapeState.mediaRecorder = new MediaRecorder(selftapeState.mediaStream, options);
    } catch (e) {
        // Fallback
        selftapeState.mediaRecorder = new MediaRecorder(selftapeState.mediaStream);
    }
    
    selftapeState.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            selftapeState.recordedChunks.push(event.data);
        }
    };
    
    selftapeState.mediaRecorder.onstop = () => {
        finishRecording();
    };
    
    // Start recording
    selftapeState.mediaRecorder.start(1000); // Collect data every second
    selftapeState.isRecording = true;
    selftapeState.recordingStartTime = Date.now();
    
    // Update UI
    document.getElementById('selftapeRecordBtn').classList.add('hidden');
    document.getElementById('selftapeStopBtn').classList.remove('hidden');
    document.getElementById('recordingIndicator').classList.remove('hidden');
    document.getElementById('framingGuide').classList.add('hidden');
    
    // Start timer
    selftapeState.recordingTimer = setInterval(updateRecordingTime, 1000);
    
    console.log('🔴 Recording started');
}

function stopSelftapeRecording() {
    if (selftapeState.mediaRecorder && selftapeState.isRecording) {
        selftapeState.mediaRecorder.stop();
        selftapeState.isRecording = false;
        
        // Clear timer
        if (selftapeState.recordingTimer) {
            clearInterval(selftapeState.recordingTimer);
            selftapeState.recordingTimer = null;
        }
        
        // Update UI
        document.getElementById('selftapeRecordBtn').classList.remove('hidden');
        document.getElementById('selftapeStopBtn').classList.add('hidden');
        document.getElementById('recordingIndicator').classList.add('hidden');
        document.getElementById('framingGuide').classList.remove('hidden');
        
        console.log('⏹ Recording stopped');
    }
}

function updateRecordingTime() {
    const elapsed = Math.floor((Date.now() - selftapeState.recordingStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const seconds = (elapsed % 60).toString().padStart(2, '0');
    document.getElementById('recTime').textContent = `${minutes}:${seconds}`;
}

function finishRecording() {
    const blob = new Blob(selftapeState.recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    
    // Create take object
    const take = {
        id: Date.now(),
        name: `Take ${selftapeState.takes.length + 1}`,
        url: url,
        blob: blob,
        duration: Math.floor((Date.now() - selftapeState.recordingStartTime) / 1000),
        timestamp: new Date().toLocaleTimeString()
    };
    
    selftapeState.takes.push(take);
    
    // Update takes grid
    updateTakesGrid();
    
    console.log(`📼 Take saved: ${take.name}`);
}

function updateTakesGrid() {
    const grid = document.getElementById('takesGrid');
    const count = document.getElementById('takesCount');
    
    if (!grid) return;
    
    count.textContent = `${selftapeState.takes.length} recording${selftapeState.takes.length !== 1 ? 's' : ''}`;
    
    if (selftapeState.takes.length === 0) {
        grid.innerHTML = `
            <div class="empty-takes">
                <span class="empty-icon">🎬</span>
                <p>No recordings yet. Start your first take!</p>
            </div>
        `;
        return;
    }
    
    grid.innerHTML = selftapeState.takes.map(take => `
        <div class="take-card" data-take-id="${take.id}">
            <div class="take-thumbnail">
                <video src="${take.url}" muted></video>
                <span class="take-duration">${formatDuration(take.duration)}</span>
            </div>
            <div class="take-info">
                <div class="take-name">${take.name}</div>
                <div class="take-time">${take.timestamp}</div>
            </div>
        </div>
    `).join('');
    
    // Add click handlers
    grid.querySelectorAll('.take-card').forEach(card => {
        card.addEventListener('click', () => {
            const takeId = parseInt(card.dataset.takeId);
            const take = selftapeState.takes.find(t => t.id === takeId);
            if (take) {
                openTakePlayback(take);
            }
        });
        
        // Generate thumbnail on hover
        const video = card.querySelector('video');
        video.addEventListener('loadeddata', () => {
            video.currentTime = 0.5; // Jump to 0.5s for thumbnail
        });
    });
}

function formatDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function openTakePlayback(take) {
    const modal = document.getElementById('playbackModal');
    const video = document.getElementById('playbackVideo');
    
    if (!modal || !video) return;
    
    video.src = take.url;
    modal.classList.remove('hidden');
    
    // Store current take ID for delete/download
    modal.dataset.currentTakeId = take.id;
}

// Self-tape event listeners
document.getElementById('selftapeRecordBtn')?.addEventListener('click', startSelftapeRecording);
document.getElementById('selftapeStopBtn')?.addEventListener('click', stopSelftapeRecording);
document.getElementById('closePlaybackBtn')?.addEventListener('click', () => {
    document.getElementById('playbackModal').classList.add('hidden');
    document.getElementById('playbackVideo').pause();
});

document.getElementById('downloadTakeBtn')?.addEventListener('click', () => {
    const modal = document.getElementById('playbackModal');
    const takeId = parseInt(modal.dataset.currentTakeId);
    const take = selftapeState.takes.find(t => t.id === takeId);
    
    if (take) {
        const a = document.createElement('a');
        a.href = take.url;
        a.download = `${take.name.replace(/\s/g, '_')}.webm`;
        a.click();
    }
});

document.getElementById('deleteTakeBtn')?.addEventListener('click', () => {
    const modal = document.getElementById('playbackModal');
    const takeId = parseInt(modal.dataset.currentTakeId);
    
    selftapeState.takes = selftapeState.takes.filter(t => t.id !== takeId);
    updateTakesGrid();
    modal.classList.add('hidden');
});

// ============================================================================
// COACH CARDS
// ============================================================================

document.querySelectorAll('.coach-card .btn-coach').forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const card = btn.closest('.coach-card');
        const coachType = card.dataset.coach;
        
        // For now, show coming soon alert
        if (coachType !== 'presentation') {
            alert(`${coachType.charAt(0).toUpperCase() + coachType.slice(1)} Coach coming soon! 🚀`);
        } else {
            // Switch to real-time tab for presentation coach
            document.querySelector('[data-tab="realtime"]').click();
        }
    });
});

console.log('🎭 Scivora initialized - Perfect Every Performance!');

// ============================================================================
// AUTO-INITIALIZE CAMERA ON PAGE LOAD
// ============================================================================

// Director's wisdom: "The mirror should always be ready for the actor"
// Initialize camera preview immediately when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Small delay to ensure DOM is ready
    setTimeout(() => {
        initCameraPreview();
    }, 500);
});

// Also try to init if DOM already loaded
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(initCameraPreview, 500);
}

// ============================================================================
// VIDEO ANALYSIS MODE
// ============================================================================

/**
 * Analyze recorded takes or uploaded videos with:
 * - Frame-by-frame playback
 * - Performance metrics timeline
 * - Director's notes at specific timestamps
 */

let analysisState = {
    currentVideo: null,
    markers: [],
    isAnalyzing: false
};

function openVideoAnalysis(videoUrl, videoName) {
    const modal = document.getElementById('playbackModal');
    const video = document.getElementById('playbackVideo');
    const analysisPanel = document.getElementById('playbackAnalysis');
    
    if (!modal || !video) return;
    
    video.src = videoUrl;
    modal.classList.remove('hidden');
    
    // Setup analysis panel
    if (analysisPanel) {
        analysisPanel.innerHTML = `
            <div class="analysis-header">
                <h3>🎬 Director's Analysis: ${videoName || 'Take'}</h3>
                <button class="btn btn-small btn-primary" id="runAnalysisBtn">
                    <span class="btn-icon">🔍</span> Analyze Performance
                </button>
            </div>
            
            <div class="analysis-timeline" id="analysisTimeline">
                <div class="timeline-track">
                    <div class="timeline-playhead" id="timelinePlayhead"></div>
                </div>
                <div class="timeline-markers" id="timelineMarkers"></div>
            </div>
            
            <div class="analysis-metrics" id="analysisMetrics">
                <div class="metric-mini">
                    <span class="metric-label">Overall</span>
                    <span class="metric-value" id="analysisOverall">--</span>
                </div>
                <div class="metric-mini">
                    <span class="metric-label">Energy</span>
                    <span class="metric-value" id="analysisEnergy">--</span>
                </div>
                <div class="metric-mini">
                    <span class="metric-label">Emotion</span>
                    <span class="metric-value" id="analysisEmotion">--</span>
                </div>
                <div class="metric-mini">
                    <span class="metric-label">Presence</span>
                    <span class="metric-value" id="analysisPresence">--</span>
                </div>
            </div>
            
            <div class="analysis-notes" id="analysisNotes">
                <h4>📝 Director's Notes</h4>
                <div class="notes-list">
                    <p class="notes-placeholder">Click "Analyze Performance" to get AI-powered feedback on this take.</p>
                </div>
            </div>
        `;
        
        // Setup analysis button
        document.getElementById('runAnalysisBtn')?.addEventListener('click', () => {
            analyzeCurrentVideo(video);
        });
        
        // Setup playhead sync
        video.addEventListener('timeupdate', () => {
            updatePlayhead(video);
        });
    }
}

function updatePlayhead(video) {
    const playhead = document.getElementById('timelinePlayhead');
    if (!playhead || !video.duration) return;
    
    const progress = (video.currentTime / video.duration) * 100;
    playhead.style.left = `${progress}%`;
}

async function analyzeCurrentVideo(video) {
    const notesDiv = document.querySelector('.analysis-notes .notes-list');
    const metrics = document.getElementById('analysisMetrics');
    
    if (notesDiv) {
        notesDiv.innerHTML = `
            <div class="analyzing-spinner">
                <span class="spinner">🎬</span>
                <span>Analyzing your performance...</span>
            </div>
        `;
    }
    
    // Simulate analysis (in production, this would send frames to the backend)
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Generate director's notes
    const notes = [
        { time: 0, type: 'positive', text: 'Strong opening presence - you command attention.' },
        { time: Math.random() * 30, type: 'tip', text: 'Try more vocal variety in this section.' },
        { time: Math.random() * 30, type: 'positive', text: 'Excellent emotional connection here.' },
        { time: Math.random() * 30, type: 'warning', text: 'Eye contact dropped - stay connected with camera.' }
    ];
    
    if (notesDiv) {
        notesDiv.innerHTML = notes.map(note => `
            <div class="note-item note-${note.type}" data-time="${note.time}">
                <span class="note-time">${formatTime(note.time)}</span>
                <span class="note-icon">${note.type === 'positive' ? '✓' : note.type === 'warning' ? '!' : '→'}</span>
                <span class="note-text">${note.text}</span>
            </div>
        `).join('');
        
        // Click to seek
        notesDiv.querySelectorAll('.note-item').forEach(item => {
            item.addEventListener('click', () => {
                video.currentTime = parseFloat(item.dataset.time);
            });
        });
    }
    
    // Update metrics
    document.getElementById('analysisOverall').textContent = Math.round(70 + Math.random() * 20);
    document.getElementById('analysisEnergy').textContent = Math.round(60 + Math.random() * 30);
    document.getElementById('analysisEmotion').textContent = '😊 Engaged';
    document.getElementById('analysisPresence').textContent = Math.round(65 + Math.random() * 25);
}

function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// Override the take playback to use our analysis mode
const originalOpenTakePlayback = openTakePlayback;
openTakePlayback = function(take) {
    openVideoAnalysis(take.url, take.name);
};

// ============================================================================
// SESSION HISTORY & ANALYSIS SYSTEM
// ============================================================================

/**
 * AI Director's Philosophy:
 * "A great performance is not born in a single take—it's sculpted through 
 * review, reflection, and refinement. Every session is a lesson."
 * 
 * This system provides:
 * 1. Persistent session storage with video recording
 * 2. Comprehensive AI Director analysis
 * 3. Progress tracking over time
 * 4. Session comparison and management
 */

// IndexedDB for video storage
const DB_NAME = 'ScivoraDB';
const DB_VERSION = 1;
const STORE_SESSIONS = 'sessions';
const STORE_VIDEOS = 'videos';

let db = null;

// Session recording state
let sessionRecording = {
    mediaRecorder: null,
    recordedChunks: [],
    isRecording: false,
    currentSession: null,
    notes: [],
    metricsHistory: []
};

// Initialize IndexedDB
async function initDatabase() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
            db = request.result;
            console.log('📦 Session database initialized');
            resolve(db);
        };
        
        request.onupgradeneeded = (event) => {
            const database = event.target.result;
            
            // Sessions store
            if (!database.objectStoreNames.contains(STORE_SESSIONS)) {
                const sessionsStore = database.createObjectStore(STORE_SESSIONS, { keyPath: 'id' });
                sessionsStore.createIndex('date', 'date', { unique: false });
            }
            
            // Videos store (for large blob data)
            if (!database.objectStoreNames.contains(STORE_VIDEOS)) {
                database.createObjectStore(STORE_VIDEOS, { keyPath: 'sessionId' });
            }
        };
    });
}

// Start recording session with video
async function startSessionRecording() {
    if (!state.mediaStream) return;
    
    sessionRecording.recordedChunks = [];
    sessionRecording.notes = [];
    sessionRecording.metricsHistory = [];
    
    // Create session ID
    const sessionId = `session_${Date.now()}`;
    sessionRecording.currentSession = {
        id: sessionId,
        date: new Date().toISOString(),
        mode: state.selectedMode,
        intensity: state.sceneIntensity,
        duration: 0,
        finalScores: null,
        transcript: '',
        notes: [],
        hasVideo: false
    };
    
    // Try to record video
    try {
        const options = { mimeType: 'video/webm;codecs=vp9,opus' };
        try {
            sessionRecording.mediaRecorder = new MediaRecorder(state.mediaStream, options);
        } catch (e) {
            sessionRecording.mediaRecorder = new MediaRecorder(state.mediaStream);
        }
        
        sessionRecording.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                sessionRecording.recordedChunks.push(event.data);
            }
        };
        
        sessionRecording.mediaRecorder.start(1000);
        sessionRecording.isRecording = true;
        sessionRecording.currentSession.hasVideo = true;
        console.log('🎥 Session video recording started');
    } catch (e) {
        console.warn('Could not start video recording:', e);
        sessionRecording.currentSession.hasVideo = false;
    }
}

// Stop recording and save session
async function stopSessionRecording() {
    if (!sessionRecording.currentSession) return;
    
    // Stop video recording
    if (sessionRecording.mediaRecorder && sessionRecording.isRecording) {
        sessionRecording.mediaRecorder.stop();
        sessionRecording.isRecording = false;
        
        // Wait for final data
        await new Promise(resolve => {
            sessionRecording.mediaRecorder.onstop = resolve;
        });
    }
    
    // Calculate duration
    sessionRecording.currentSession.duration = state.sessionStartTime 
        ? Math.floor((Date.now() - state.sessionStartTime) / 1000)
        : 0;
    
    // Save transcript
    sessionRecording.currentSession.transcript = state.fullTranscript;
    sessionRecording.currentSession.notes = [...sessionRecording.notes];
    sessionRecording.currentSession.metricsHistory = [...sessionRecording.metricsHistory];
    
    // Save to database
    await saveSession(sessionRecording.currentSession, sessionRecording.recordedChunks);
    
    console.log('💾 Session saved:', sessionRecording.currentSession.id);
    
    // Generate AI Director analysis
    generateDirectorAnalysis(sessionRecording.currentSession);
}

// Save session to IndexedDB
async function saveSession(session, videoChunks) {
    if (!db) await initDatabase();
    
    // Prepare video data BEFORE transaction (to avoid transaction timeout)
    let videoData = null;
    if (videoChunks && videoChunks.length > 0) {
        const videoBlob = new Blob(videoChunks, { type: 'video/webm' });
        const thumbnail = await generateThumbnail(videoBlob);
        videoData = {
            sessionId: session.id,
            video: videoBlob,
            thumbnail: thumbnail
        };
    }
    
    // Now do the actual database save in one transaction
    return new Promise((resolve, reject) => {
        const tx = db.transaction([STORE_SESSIONS, STORE_VIDEOS], 'readwrite');
        
        tx.oncomplete = () => {
            console.log('💾 Session saved:', session.id);
            loadSessionHistory();
            resolve();
        };
        
        tx.onerror = (e) => {
            console.error('Failed to save session:', e);
            reject(e);
        };
        
        // Save session
        tx.objectStore(STORE_SESSIONS).put(session);
        
        // Save video if available
        if (videoData) {
            tx.objectStore(STORE_VIDEOS).put(videoData);
        }
    });
}

// Generate thumbnail from video
async function generateThumbnail(videoBlob) {
    return new Promise((resolve) => {
        const video = document.createElement('video');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        video.onloadeddata = () => {
            video.currentTime = 1; // Jump to 1 second
        };
        
        video.onseeked = () => {
            canvas.width = 320;
            canvas.height = 180;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            resolve(canvas.toDataURL('image/jpeg', 0.7));
            URL.revokeObjectURL(video.src);
        };
        
        video.onerror = () => resolve(null);
        video.src = URL.createObjectURL(videoBlob);
    });
}

// Record note during session
function recordSessionNote(note) {
    if (!sessionRecording.currentSession) return;
    
    const timestamp = state.sessionStartTime 
        ? Math.floor((Date.now() - state.sessionStartTime) / 1000)
        : 0;
    
    sessionRecording.notes.push({
        time: timestamp,
        type: note.type || 'tip',
        text: note.message || note.tip || note.text || ''
    });
}

// Record metrics snapshot
function recordMetricsSnapshot(metrics) {
    if (!sessionRecording.currentSession) return;
    
    const timestamp = state.sessionStartTime 
        ? Math.floor((Date.now() - state.sessionStartTime) / 1000)
        : 0;
    
    sessionRecording.metricsHistory.push({
        time: timestamp,
        scores: { ...metrics.scores },
        raw: { ...metrics.raw }
    });
    
    // Update final scores
    sessionRecording.currentSession.finalScores = { ...metrics.scores };
}

// Load session history
async function loadSessionHistory() {
    if (!db) await initDatabase();
    
    try {
        // Get all sessions in one transaction
        const sessions = await new Promise((resolve, reject) => {
            const tx = db.transaction([STORE_SESSIONS], 'readonly');
            const request = tx.objectStore(STORE_SESSIONS).getAll();
            request.onsuccess = () => resolve(request.result || []);
            request.onerror = () => reject(request.error);
        });
        
        // Get all videos in a separate transaction
        const videos = await new Promise((resolve, reject) => {
            const tx = db.transaction([STORE_VIDEOS], 'readonly');
            const request = tx.objectStore(STORE_VIDEOS).getAll();
            request.onsuccess = () => resolve(request.result || []);
            request.onerror = () => reject(request.error);
        });
        
        // Create a map of video thumbnails
        const thumbnailMap = {};
        videos.forEach(v => {
            if (v?.sessionId && v?.thumbnail) {
                thumbnailMap[v.sessionId] = v.thumbnail;
            }
        });
        
        // Sort by date (newest first)
        sessions.sort((a, b) => new Date(b.date) - new Date(a.date));
        
        console.log(`📊 Loaded ${sessions.length} sessions, ${videos.length} videos`);
        
        // Update stats
        updateHistoryStats(sessions);
        
        // Render sessions with thumbnail map
        renderSessionsGrid(sessions, thumbnailMap);
        
        // Update progress chart
        updateProgressChart(sessions);
    } catch (error) {
        console.error('Failed to load session history:', error);
    }
}

// Update history statistics
function updateHistoryStats(sessions) {
    const totalSessions = sessions.length;
    const totalTime = sessions.reduce((sum, s) => sum + (s.duration || 0), 0);
    const avgScore = sessions.length > 0
        ? Math.round(sessions.reduce((sum, s) => sum + ((s.finalScores?.overall || 0.5) * 100), 0) / sessions.length)
        : '--';
    
    // Calculate improvement (compare first 3 vs last 3 sessions)
    let improvement = '--';
    if (sessions.length >= 6) {
        const recent = sessions.slice(0, 3);
        const older = sessions.slice(-3);
        const recentAvg = recent.reduce((sum, s) => sum + ((s.finalScores?.overall || 0.5) * 100), 0) / 3;
        const olderAvg = older.reduce((sum, s) => sum + ((s.finalScores?.overall || 0.5) * 100), 0) / 3;
        const diff = recentAvg - olderAvg;
        improvement = diff >= 0 ? `+${Math.round(diff)}%` : `${Math.round(diff)}%`;
    }
    
    // Update DOM
    document.getElementById('totalSessions').textContent = totalSessions;
    document.getElementById('totalTime').textContent = formatDurationLong(totalTime);
    document.getElementById('avgScore').textContent = avgScore;
    document.getElementById('improvement').textContent = improvement;
}

// Format duration for display
function formatDurationLong(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
}

// Render sessions grid
function renderSessionsGrid(sessions, thumbnailMap) {
    const grid = document.getElementById('sessionsGrid');
    const empty = document.getElementById('emptySessions');
    
    if (!grid) return;
    
    if (sessions.length === 0) {
        grid.innerHTML = '';
        if (empty) {
            grid.appendChild(empty);
            empty.style.display = 'block';
        }
        return;
    }
    
    if (empty) empty.style.display = 'none';
    
    grid.innerHTML = sessions.map(session => {
        const score = Math.round((session.finalScores?.overall || 0.5) * 100);
        const scoreClass = score >= 70 ? 'score-high' : score >= 50 ? 'score-medium' : 'score-low';
        const date = new Date(session.date);
        const dateStr = date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
        const thumbnail = thumbnailMap[session.id];
        const previewNote = session.notes?.[0]?.text || 'No notes recorded';
        
        return `
            <div class="session-card" data-session-id="${session.id}">
                <div class="session-thumbnail">
                    ${thumbnail 
                        ? `<img src="${thumbnail}" alt="Session thumbnail">`
                        : `<div class="no-video">🎭</div>`
                    }
                    <span class="session-duration">${formatDuration(session.duration || 0)}</span>
                    <span class="session-score-badge ${scoreClass}">${score}</span>
                </div>
                <div class="session-info">
                    <div class="session-date">${dateStr}</div>
                    <div class="session-meta">
                        <span>🎬 ${session.mode?.replace('_', ' ') || 'Free Practice'}</span>
                        <span>📝 ${(session.transcript?.split(' ').length || 0)} words</span>
                    </div>
                    <div class="session-preview-notes">${previewNote}</div>
                </div>
            </div>
        `;
    }).join('');
    
    // Add click handlers
    grid.querySelectorAll('.session-card').forEach(card => {
        card.addEventListener('click', () => {
            const sessionId = card.dataset.sessionId;
            openSessionDetail(sessionId);
        });
    });
}

// Update progress chart
function updateProgressChart(sessions) {
    const placeholder = document.getElementById('chartPlaceholder');
    const canvas = document.getElementById('progressCanvas');
    
    if (sessions.length < 2) {
        placeholder.classList.remove('hidden');
        return;
    }
    
    placeholder.classList.add('hidden');
    
    // Simple canvas chart
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    // Get last 10 sessions
    const data = sessions.slice(0, 10).reverse().map(s => ({
        score: (s.finalScores?.overall || 0.5) * 100,
        date: new Date(s.date)
    }));
    
    // Draw chart
    const padding = 40;
    const chartWidth = canvas.width - padding * 2;
    const chartHeight = canvas.height - padding * 2;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.fillRect(padding, padding, chartWidth, chartHeight);
    
    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding + (chartHeight / 4) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(canvas.width - padding, y);
        ctx.stroke();
    }
    
    // Data line
    ctx.strokeStyle = '#d4af37';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    data.forEach((point, i) => {
        const x = padding + (chartWidth / (data.length - 1)) * i;
        const y = padding + chartHeight - (point.score / 100) * chartHeight;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
    
    // Data points
    ctx.fillStyle = '#d4af37';
    data.forEach((point, i) => {
        const x = padding + (chartWidth / (data.length - 1)) * i;
        const y = padding + chartHeight - (point.score / 100) * chartHeight;
        
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
    });
}

// Open session detail modal
async function openSessionDetail(sessionId) {
    if (!db) await initDatabase();
    
    const modal = document.getElementById('sessionDetailModal');
    const video = document.getElementById('sessionVideo');
    const noVideo = document.getElementById('videoNoRecording');
    
    // Get session data
    const tx = db.transaction([STORE_SESSIONS, STORE_VIDEOS], 'readonly');
    const session = await new Promise((resolve, reject) => {
        const request = tx.objectStore(STORE_SESSIONS).get(sessionId);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
    
    const videoData = await new Promise((resolve, reject) => {
        const request = tx.objectStore(STORE_VIDEOS).get(sessionId);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
    
    if (!session) return;
    
    // Set modal data
    modal.dataset.sessionId = sessionId;
    
    // Update title
    const date = new Date(session.date);
    document.getElementById('sessionDetailTitle').textContent = 
        `📊 Session: ${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`;
    
    // Video
    if (videoData?.video) {
        video.src = URL.createObjectURL(videoData.video);
        video.classList.remove('hidden');
        noVideo.classList.add('hidden');
    } else {
        video.classList.add('hidden');
        noVideo.classList.remove('hidden');
    }
    
    // Score
    const score = Math.round((session.finalScores?.overall || 0.5) * 100);
    document.getElementById('detailScore').textContent = score;
    
    // Score breakdown
    const breakdown = document.getElementById('detailBreakdown');
    if (session.finalScores) {
        breakdown.innerHTML = Object.entries(session.finalScores)
            .filter(([key]) => key !== 'overall')
            .map(([key, value]) => `
                <div class="breakdown-item">
                    <span class="label">${key.replace('_', ' ')}</span>
                    <span class="value">${Math.round(value * 100)}</span>
                </div>
            `).join('');
    }
    
    // Session notes
    const notesList = document.getElementById('sessionNotesList');
    if (session.notes && session.notes.length > 0) {
        notesList.innerHTML = session.notes.map(note => `
            <div class="session-note-item ${note.type}">
                <span class="note-time">${formatTime(note.time)}</span>
                <span class="note-text">${note.text}</span>
            </div>
        `).join('');
    } else {
        notesList.innerHTML = '<p style="color: var(--text-muted); font-style: italic;">No notes recorded for this session.</p>';
    }
    
    // Transcript
    const transcriptEl = document.getElementById('sessionTranscript');
    if (session.transcript) {
        transcriptEl.innerHTML = formatTranscript(session.transcript);
    } else {
        transcriptEl.innerHTML = '<em style="color: var(--text-muted);">No transcript available.</em>';
    }
    
    // Timeline markers
    renderTimelineMarkers(session);
    
    // Generate AI Director's Analysis
    generateDirectorAnalysis(session);
    
    // Show modal
    modal.classList.remove('hidden');
}

// Render timeline markers
function renderTimelineMarkers(session) {
    const container = document.getElementById('sessionTimelineMarkers');
    if (!container || !session.notes) return;
    
    const duration = session.duration || 1;
    
    container.innerHTML = session.notes.map(note => {
        const position = (note.time / duration) * 100;
        return `<div class="timeline-marker ${note.type}" style="left: ${position}%" title="${note.text}"></div>`;
    }).join('');
}

// Generate AI Director's Analysis
async function generateDirectorAnalysis(session) {
    const reportEl = document.getElementById('directorReport');
    if (!reportEl) return;
    
    // Show loading
    reportEl.innerHTML = `
        <div class="report-loading">
            <span class="loading-spinner">🎬</span>
            <p>The Director is reviewing your performance...</p>
        </div>
    `;
    
    // Analyze the session
    const analysis = analyzePerformance(session);
    
    // Simulate AI thinking time for dramatic effect
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Render the report
    reportEl.innerHTML = `
        <div class="report-section">
            <h4>🎯 Overall Assessment</h4>
            <p>${analysis.overall}</p>
        </div>
        
        <div class="report-section">
            <h4>✨ What Worked Well</h4>
            <ul>
                ${analysis.strengths.map(s => `<li>${s}</li>`).join('')}
            </ul>
        </div>
        
        <div class="report-section">
            <h4>🎯 Areas for Improvement</h4>
            <ul>
                ${analysis.improvements.map(i => `<li>${i}</li>`).join('')}
            </ul>
        </div>
        
        <div class="report-section">
            <h4>📝 Director's Notes</h4>
            <p>${analysis.directorNote}</p>
        </div>
        
        <div class="report-section">
            <h4>🎬 For Your Next Session</h4>
            <ul>
                ${analysis.nextSteps.map(n => `<li>${n}</li>`).join('')}
            </ul>
        </div>
    `;
}

// Analyze performance and generate insights
function analyzePerformance(session) {
    const scores = session.finalScores || {};
    const overall = (scores.overall || 0.5) * 100;
    const pace = (scores.pace || 0.5) * 100;
    const energy = (scores.energy || 0.5) * 100;
    const eyeContact = (scores.eye_contact || 0.5) * 100;
    const fillers = (scores.filler_words || 0.5) * 100;
    const variety = (scores.pitch_variety || 0.5) * 100;
    const presence = (scores.presence || 0.5) * 100;
    
    const strengths = [];
    const improvements = [];
    
    // Analyze each metric
    if (eyeContact >= 70) {
        strengths.push("Excellent eye contact - you connected well with your audience");
    } else if (eyeContact < 50) {
        improvements.push("Work on maintaining consistent eye contact with the camera");
    }
    
    if (pace >= 60 && pace <= 80) {
        strengths.push("Your pacing was well-controlled and natural");
    } else if (pace < 50) {
        improvements.push("Try to slow down - give your words room to breathe");
    } else if (pace > 80) {
        improvements.push("Consider using more pauses for dramatic effect");
    }
    
    if (energy >= 70) {
        strengths.push("Strong vocal energy and projection");
    } else if (energy < 50) {
        improvements.push("Project your voice more - fill the space with your energy");
    }
    
    if (variety >= 70) {
        strengths.push("Great vocal variety - your delivery was dynamic and engaging");
    } else if (variety < 50) {
        improvements.push("Add more vocal variety - vary your pitch and tone");
    }
    
    if (fillers >= 80) {
        strengths.push("Clean delivery with minimal filler words");
    } else if (fillers < 60) {
        improvements.push("Work on eliminating filler words (um, uh, like)");
    }
    
    if (presence >= 70) {
        strengths.push("Commanding screen presence - you owned the moment");
    } else if (presence < 50) {
        improvements.push("Focus on your overall presence - commit fully to each moment");
    }
    
    // Generate overall assessment
    let overallText;
    if (overall >= 80) {
        overallText = "An outstanding performance! You demonstrated excellent control across all areas. This is the kind of work that books jobs. Keep this energy and continue to refine.";
    } else if (overall >= 65) {
        overallText = "A solid performance with clear strengths. You're on the right track. Focus on the areas noted below to elevate your work to the next level.";
    } else if (overall >= 50) {
        overallText = "A decent foundation with room for growth. The potential is there - now it's about consistent practice and attention to the fundamentals.";
    } else {
        overallText = "Every great actor started somewhere. This session gives us clear areas to work on. Focus on one thing at a time, and you'll see improvement quickly.";
    }
    
    // Director's note based on session content
    const directorNotes = [
        "Remember: the camera sees everything. Every micro-expression, every moment of truth or falsehood. Commit fully to your choices.",
        "The best performances come from listening, not just waiting for your turn to speak. Even in a monologue, you're listening to yourself, to the space, to the imagined other.",
        "Technique is the foundation, but don't let it cage you. Once you've mastered the rules, you earn the right to break them.",
        "Your unique perspective is your greatest asset. Don't try to be someone else - bring yourself fully to the character.",
        "Tension is the enemy of good acting. Find the places where you're holding and learn to release. The body tells the truth."
    ];
    
    // Next steps
    const nextSteps = [];
    if (improvements.length > 0) {
        nextSteps.push(`Focus on ${improvements[0].toLowerCase().split(' - ')[0]}`);
    }
    nextSteps.push("Try the same scene at a different intensity level");
    if (eyeContact < 70) {
        nextSteps.push("Practice a scene focused purely on maintaining camera connection");
    }
    if (variety < 60) {
        nextSteps.push("Read a children's book out loud - exaggerate every emotion");
    }
    nextSteps.push("Record yourself and watch the playback critically");
    
    return {
        overall: overallText,
        strengths: strengths.length > 0 ? strengths : ["You showed up and did the work - that's the first step"],
        improvements: improvements.length > 0 ? improvements : ["Keep practicing to find your unique voice"],
        directorNote: directorNotes[Math.floor(Math.random() * directorNotes.length)],
        nextSteps: nextSteps.slice(0, 4)
    };
}

// Delete session
async function deleteSession(sessionId) {
    if (!db) await initDatabase();
    
    if (!confirm('Are you sure you want to delete this session? This cannot be undone.')) {
        return;
    }
    
    const tx = db.transaction([STORE_SESSIONS, STORE_VIDEOS], 'readwrite');
    await tx.objectStore(STORE_SESSIONS).delete(sessionId);
    await tx.objectStore(STORE_VIDEOS).delete(sessionId);
    
    // Close modal and reload
    document.getElementById('sessionDetailModal').classList.add('hidden');
    loadSessionHistory();
    
    console.log('🗑️ Session deleted:', sessionId);
}

// Clear all sessions
async function clearAllSessions() {
    if (!db) await initDatabase();
    
    if (!confirm('Are you sure you want to delete ALL sessions? This cannot be undone.')) {
        return;
    }
    
    const tx = db.transaction([STORE_SESSIONS, STORE_VIDEOS], 'readwrite');
    await tx.objectStore(STORE_SESSIONS).clear();
    await tx.objectStore(STORE_VIDEOS).clear();
    
    loadSessionHistory();
    console.log('🗑️ All sessions cleared');
}

// Export session data
async function exportSession(sessionId) {
    if (!db) await initDatabase();
    
    const tx = db.transaction([STORE_SESSIONS, STORE_VIDEOS], 'readonly');
    const session = await new Promise((resolve, reject) => {
        const request = tx.objectStore(STORE_SESSIONS).get(sessionId);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
    
    const videoData = await new Promise((resolve, reject) => {
        const request = tx.objectStore(STORE_VIDEOS).get(sessionId);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
    
    // Export video if available
    if (videoData?.video) {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(videoData.video);
        a.download = `scivora_session_${new Date(session.date).toISOString().split('T')[0]}.webm`;
        a.click();
    }
    
    // Export data as JSON
    const exportData = {
        ...session,
        exportedAt: new Date().toISOString()
    };
    
    const jsonBlob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(jsonBlob);
    a.download = `scivora_session_${new Date(session.date).toISOString().split('T')[0]}.json`;
    a.click();
}

// Session recording hooks are now integrated directly into startSession/stopSession

// Event listeners for session history
document.getElementById('closeSessionDetailBtn')?.addEventListener('click', () => {
    document.getElementById('sessionDetailModal').classList.add('hidden');
    const video = document.getElementById('sessionVideo');
    if (video) {
        video.pause();
        video.src = '';
    }
});

document.getElementById('deleteSessionBtn')?.addEventListener('click', () => {
    const sessionId = document.getElementById('sessionDetailModal').dataset.sessionId;
    if (sessionId) deleteSession(sessionId);
});

document.getElementById('downloadSessionBtn')?.addEventListener('click', () => {
    const sessionId = document.getElementById('sessionDetailModal').dataset.sessionId;
    if (sessionId) exportSession(sessionId);
});

document.getElementById('reanalyzeSessionBtn')?.addEventListener('click', async () => {
    const sessionId = document.getElementById('sessionDetailModal').dataset.sessionId;
    if (!sessionId || !db) return;
    
    const session = await new Promise((resolve, reject) => {
        const request = db.transaction(STORE_SESSIONS).objectStore(STORE_SESSIONS).get(sessionId);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
    
    if (session) generateDirectorAnalysis(session);
});

document.getElementById('clearSessionsBtn')?.addEventListener('click', clearAllSessions);

document.getElementById('exportSessionsBtn')?.addEventListener('click', async () => {
    if (!db) await initDatabase();
    
    const sessions = await new Promise((resolve, reject) => {
        const request = db.transaction(STORE_SESSIONS).objectStore(STORE_SESSIONS).getAll();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
    
    const exportData = {
        exported: new Date().toISOString(),
        sessions: sessions
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `scivora_all_sessions_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
});

// Chart filter buttons
document.querySelectorAll('.filter-btn[data-range]').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn[data-range]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        // TODO: Implement date range filtering
    });
});

// Initialize database and load history on page load
initDatabase().then(() => {
    loadSessionHistory();
    console.log('📊 Session history loaded');
});

// ============================================================================
// SCRIPT TELEPROMPTER PANEL
// ============================================================================

/**
 * Sample scripts for practice - organized by category and intensity
 */
const SAMPLE_SCRIPTS = {
    monologues: {
        intimate: [
            {
                title: "The Letter",
                context: "A person reading a letter from someone they've lost",
                script: `I found your letter today. The one you wrote before... before everything changed.

<span class="stage-direction">(pause)</span>

You said you'd always be here. That no matter what happened, we'd figure it out together.

<span class="stage-direction">(softer)</span>

I believed you. I still believe you.

<span class="stage-direction">(long pause)</span>

I just wish... I wish I had one more day. One more chance to tell you everything I never said.

<span class="stage-direction">(whisper)</span>

I miss you.`,
                tips: [
                    "Keep your voice soft and intimate",
                    "Let the pauses breathe - silence is powerful",
                    "Eye contact should feel like speaking to someone close",
                    "Build emotion gradually - don't start at the climax"
                ]
            },
            {
                title: "The Apology",
                context: "Someone apologizing to a loved one after a betrayal",
                script: `I know you don't want to hear this. And I know... I know I have no right to ask for anything.

<span class="stage-direction">(beat)</span>

What I did was wrong. There's no excuse. No explanation that makes it okay.

I could tell you I was scared. That I didn't know what else to do. But that would just be more lies.

<span class="stage-direction">(voice breaking)</span>

The truth is, I was selfish. I thought only about myself. And I hurt the one person who never deserved it.

<span class="stage-direction">(quiet)</span>

I'm not asking you to forgive me. I just needed you to know... I'm sorry. I am so, so sorry.`,
                tips: [
                    "Vulnerability is key - let us see your shame",
                    "Don't push for tears - let them come naturally",
                    "The pauses are where the real acting happens",
                    "Your body language should match your words"
                ]
            }
        ],
        neutral: [
            {
                title: "The Interview",
                context: "Someone explaining their life story in a job interview",
                script: `When I was twelve, my father lost his job. We had to move three times in two years.

I learned something important during that time: adaptability isn't just a skill - it's survival.

Every new school, every new neighborhood, I had to figure out how to fit in. How to make friends. How to start over.

Some people would call that unstable. I call it training.

Because here's the thing - every company, every team, every project... they all have their own culture. Their own rhythm.

And I've spent my whole life learning how to read a room. How to adapt. How to contribute from day one.

That's not on my resume. But it's the most valuable thing I bring to this table.`,
                tips: [
                    "Confident but not arrogant",
                    "Make it conversational, not rehearsed",
                    "Find the personal connection to the material",
                    "Let your natural charm come through"
                ]
            },
            {
                title: "The Explanation",
                context: "A teacher explaining a complex concept simply",
                script: `Okay, so imagine you're standing in a river.

The water is flowing past you, right? You can feel it pushing against your legs.

Now, that water - it was somewhere else before it reached you. And after it passes you, it'll be somewhere else again.

Time is like that river.

We stand in the present moment, feeling it flow around us. The past already flowed by. The future is still upstream.

But here's the beautiful part - unlike a river, we're not just standing still. We're making choices. We're changing the direction of the current.

Every decision you make... it's like dropping a stone into that river. The ripples spread out in all directions.

Does that make sense?`,
                tips: [
                    "Speak clearly and at a measured pace",
                    "Use your hands to illustrate concepts",
                    "Check for understanding - watch your 'student'",
                    "Enthusiasm for the subject is contagious"
                ]
            }
        ],
        heightened: [
            {
                title: "The Goodbye",
                context: "Saying goodbye to someone at the airport, possibly forever",
                script: `I keep telling myself this isn't the end. That we'll see each other again.

<span class="stage-direction">(fighting tears)</span>

But standing here, watching you walk toward that gate...

I don't know anymore. I don't know if I'm strong enough to do this.

<span class="stage-direction">(grabs their hand)</span>

Promise me something. Promise me that no matter what happens, no matter how far apart we are... you won't forget what we had.

<span class="stage-direction">(voice breaking)</span>

Because I won't. I couldn't. Not if I tried.

<span class="stage-direction">(pulling them close)</span>

I love you. I will always love you. And that... that has to be enough.`,
                tips: [
                    "Let the emotion build naturally",
                    "Don't be afraid to cry, but don't force it",
                    "Physical contact creates powerful moments",
                    "The goodbye itself is the climax - everything builds to it"
                ]
            },
            {
                title: "The Confession",
                context: "Someone finally admitting a secret they've kept for years",
                script: `There's something I need to tell you. Something I should have told you a long time ago.

<span class="stage-direction">(deep breath)</span>

The night of the accident... I wasn't where I said I was.

I know. I know what you're thinking. All these years, all this time...

<span class="stage-direction">(breaking down)</span>

I was there. I was at the party. I saw what happened. I saw everything.

And I ran. I ran because I was scared. Because I was a coward.

<span class="stage-direction">(sobbing)</span>

I let you blame yourself. I let you carry that guilt for years. And I... I couldn't live with myself anymore.

I'm so sorry. God, I'm so sorry.`,
                tips: [
                    "Start controlled, lose control as you reveal more",
                    "Shame is the underlying emotion throughout",
                    "Don't rush - each revelation needs space",
                    "Let us see the weight lift as you confess"
                ]
            }
        ],
        intense: [
            {
                title: "The Confrontation",
                context: "Finally standing up to someone who has wronged you",
                script: `No. No, you don't get to do this anymore.

<span class="stage-direction">(stepping forward)</span>

You don't get to walk in here, say whatever you want, and expect me to just... take it.

I spent years trying to be what you wanted. Years bending myself into shapes that would make you happy. And you know what? It was never enough.

<span class="stage-direction">(building anger)</span>

Nothing I did was ever good enough for you!

<span class="stage-direction">(pointing)</span>

You made me feel small. You made me feel worthless. You made me question everything I am!

<span class="stage-direction">(firm, controlled)</span>

Well, not anymore. I see you now. I see exactly who you are. And I am DONE.`,
                tips: [
                    "Channel your real frustrations into this",
                    "Start from a place of controlled anger",
                    "The explosion should feel earned, not sudden",
                    "End with power, not hysteria"
                ]
            },
            {
                title: "The Defense",
                context: "A closing argument in a life-or-death case",
                script: `Ladies and gentlemen of the jury, you've heard the evidence. You've heard the testimony.

Now I'm asking you to do something harder than believing what you've been told. I'm asking you to think.

<span class="stage-direction">(walking toward jury)</span>

Because a man's life hangs in the balance. And once you make this decision, there's no taking it back.

<span class="stage-direction">(building intensity)</span>

The prosecution wants you to believe this is simple. Black and white. Open and shut.

But life isn't simple, is it? People aren't simple.

<span class="stage-direction">(pointing at defendant)</span>

That man right there - he has a family. He has children who are waiting for him to come home.

<span class="stage-direction">(passionate)</span>

Do NOT let them down because it's easier to convict than to question. Do NOT send an innocent man to his death because you're afraid to demand more evidence.

<span class="stage-direction">(quiet intensity)</span>

Do the right thing. I'm begging you. Do the right thing.`,
                tips: [
                    "Vary your intensity - not all loud, not all quiet",
                    "Make eye contact with different 'jurors'",
                    "Physical movement emphasizes key points",
                    "The final appeal should land like a punch"
                ]
            }
        ]
    },
    scenes: {
        intimate: [
            {
                title: "The Hospital",
                context: "Visiting someone who is seriously ill",
                script: `<span class="character-name">ALEX</span>
<span class="dialogue">Hey. I brought you these. They're from your garden. Sarah's been watering them.</span>

<span class="character-name">JORDAN</span>
<span class="dialogue">They're beautiful. Thank you.</span>

<span class="character-name">ALEX</span>
<span class="dialogue">How are you feeling today?</span>

<span class="character-name">JORDAN</span>
<span class="dialogue">Tired. Always tired now. But better now that you're here.</span>

<span class="character-name">ALEX</span>
<span class="dialogue">I should have come sooner. I just... I didn't know what to say.</span>

<span class="character-name">JORDAN</span>
<span class="dialogue">You don't have to say anything. Just... stay with me for a while?</span>

<span class="character-name">ALEX</span>
<span class="dialogue">I'm not going anywhere. Not ever again.</span>`,
                tips: [
                    "The unspoken words are as important as the spoken ones",
                    "Keep physical distance meaningful",
                    "Let silences happen naturally",
                    "Connect genuinely with your scene partner"
                ]
            }
        ],
        neutral: [
            {
                title: "The Roommates",
                context: "Two roommates discussing apartment rules",
                script: `<span class="character-name">CASEY</span>
<span class="dialogue">Okay, we need to talk about the dishes.</span>

<span class="character-name">MORGAN</span>
<span class="dialogue">What about them?</span>

<span class="character-name">CASEY</span>
<span class="dialogue">There's a sink full of them. Have been for three days.</span>

<span class="character-name">MORGAN</span>
<span class="dialogue">They're soaking.</span>

<span class="character-name">CASEY</span>
<span class="dialogue">For THREE days?</span>

<span class="character-name">MORGAN</span>
<span class="dialogue">Look, I've been really busy with work. I'll get to them.</span>

<span class="character-name">CASEY</span>
<span class="dialogue">That's what you said last week. And the week before.</span>

<span class="character-name">MORGAN</span>
<span class="dialogue">Fine. I'll do them tonight. Happy?</span>

<span class="character-name">CASEY</span>
<span class="dialogue">Ecstatic.</span>`,
                tips: [
                    "Find the humor in the frustration",
                    "Keep it real - this is everyday life",
                    "React to what your partner gives you",
                    "The subtext is about respect, not dishes"
                ]
            }
        ],
        heightened: [
            {
                title: "The Breakup",
                context: "A couple ending a long relationship",
                script: `<span class="character-name">JAMIE</span>
<span class="dialogue">I think we need to talk.</span>

<span class="character-name">SAM</span>
<span class="dialogue">That's never a good sentence.</span>

<span class="character-name">JAMIE</span>
<span class="dialogue">I've been doing a lot of thinking. About us. About where we're going.</span>

<span class="character-name">SAM</span>
<span class="dialogue">And?</span>

<span class="character-name">JAMIE</span>
<span class="dialogue">I don't think... I don't think we want the same things anymore.</span>

<span class="character-name">SAM</span>
<span class="dialogue">What are you saying?</span>

<span class="character-name">JAMIE</span>
<span class="dialogue">I'm saying... maybe we should take a break.</span>

<span class="character-name">SAM</span>
<span class="dialogue">A break. After four years, you want a "break"?</span>

<span class="character-name">JAMIE</span>
<span class="dialogue">Please don't make this harder than it already is.</span>

<span class="character-name">SAM</span>
<span class="dialogue">I didn't make it anything. This is all you.</span>`,
                tips: [
                    "Both characters are in pain - show it differently",
                    "Avoid playing the villain",
                    "Let the history of the relationship inform your choices",
                    "The silences are loaded with years of memories"
                ]
            }
        ],
        intense: [
            {
                title: "The Truth",
                context: "Confronting a friend about a betrayal",
                script: `<span class="character-name">RILEY</span>
<span class="dialogue">You knew. This whole time, you knew.</span>

<span class="character-name">TAYLOR</span>
<span class="dialogue">Please, let me explain—</span>

<span class="character-name">RILEY</span>
<span class="dialogue">Explain what? How you lied to my face for months?</span>

<span class="character-name">TAYLOR</span>
<span class="dialogue">I was trying to protect you!</span>

<span class="character-name">RILEY</span>
<span class="dialogue">Protect me? By letting me walk around like an idiot while everyone else knew?</span>

<span class="character-name">TAYLOR</span>
<span class="dialogue">I didn't know how to tell you. I didn't want to be the one to break your heart.</span>

<span class="character-name">RILEY</span>
<span class="dialogue">Well congratulations. You just broke something else. Our friendship.</span>

<span class="character-name">TAYLOR</span>
<span class="dialogue">Riley, please—</span>

<span class="character-name">RILEY</span>
<span class="dialogue">Don't. Just... don't.</span>`,
                tips: [
                    "Betrayal cuts deeper than anger",
                    "The relationship history matters",
                    "Let us see the hurt beneath the anger",
                    "The person being confronted has their own valid perspective"
                ]
            }
        ]
    },
    audition: {
        neutral: [
            {
                title: "The Detective",
                context: "Police detective interviewing a witness",
                script: `<span class="character-name">DETECTIVE</span>
<span class="dialogue">Can you walk me through what happened that night? Take your time.</span>

<span class="stage-direction">(listens)</span>

And what time was this, approximately?

<span class="stage-direction">(writing)</span>

Did you notice anything unusual? Anyone who seemed out of place?

<span class="stage-direction">(looking up)</span>

I know this is difficult. But anything you can remember, even small details, could help us.

<span class="stage-direction">(leaning in)</span>

What about sounds? Did you hear anything?

<span class="stage-direction">(nodding)</span>

That's helpful. Really helpful. Is there anything else? Anything at all?`,
                tips: [
                    "Maintain professional composure throughout",
                    "React to the unseen witness's responses",
                    "Balance authority with empathy",
                    "Show you're actively processing information"
                ]
            },
            {
                title: "The Doctor",
                context: "Delivering news to a patient's family",
                script: `<span class="character-name">DOCTOR</span>
<span class="dialogue">Please, have a seat.</span>

<span class="stage-direction">(sits across from them)</span>

The surgery went well. Better than expected, actually.

<span class="stage-direction">(pause)</span>

But there were some complications we didn't anticipate.

<span class="stage-direction">(gentle)</span>

Your father is stable now, but the next 48 hours are critical. We're monitoring him closely.

<span class="stage-direction">(compassionate)</span>

I know this isn't the news you were hoping for. But I want you to know, we're doing everything we can.

Do you have any questions?`,
                tips: [
                    "Balance professionalism with humanity",
                    "Choose your words carefully - they carry weight",
                    "Give space for the family to process",
                    "Show competence without arrogance"
                ]
            }
        ],
        intense: [
            {
                title: "The Witness",
                context: "Testifying about a crime you witnessed",
                script: `<span class="character-name">WITNESS</span>
<span class="dialogue">Yes, I was there. I saw everything.</span>

<span class="stage-direction">(visibly shaken)</span>

It happened so fast. One minute everything was normal, and then...

<span class="stage-direction">(deep breath)</span>

He had a gun. He was pointing it at her. I... I couldn't move. I wanted to help but I couldn't move.

<span class="stage-direction">(tears)</span>

And then the shot. That sound... I'll never forget that sound.

<span class="stage-direction">(pointing at defendant)</span>

It was him. I'll never forget his face. He looked right at me before he ran.`,
                tips: [
                    "Reliving the trauma is the key to authenticity",
                    "Physical reactions tell the story",
                    "The identification should be definitive",
                    "Don't hold back on the emotional memory"
                ]
            }
        ]
    },
    commercial: {
        neutral: [
            {
                title: "The Car Commercial",
                context: "Talking to camera about a new car",
                script: `You know what I love about Sunday mornings?

<span class="stage-direction">(looking out)</span>

That moment when you're driving down an empty road, coffee in the cupholder, nowhere you have to be...

<span class="stage-direction">(smiling)</span>

That's freedom.

<span class="stage-direction">(looks at camera)</span>

The all-new Horizon. Starting at $32,000.

Because some moments are worth driving for.`,
                tips: [
                    "Authentic enthusiasm, not salesperson energy",
                    "Create genuine emotional connection",
                    "The product reveal should feel natural",
                    "Warmth, not hype"
                ]
            },
            {
                title: "The Medicine Commercial",
                context: "Talking about life with a medical condition",
                script: `Three months ago, I couldn't make it through the day without pain.

Simple things - playing with my grandkids, taking a walk - felt impossible.

<span class="stage-direction">(hopeful)</span>

Then my doctor told me about this treatment.

<span class="stage-direction">(brighter)</span>

Now? I'm back to doing what I love. I can keep up with them again.

<span class="stage-direction">(sincere)</span>

If you're struggling like I was, talk to your doctor. It changed my life.`,
                tips: [
                    "Real vulnerability, not pity",
                    "The transformation should feel earned",
                    "Speak to someone in the same situation",
                    "Hope without false promises"
                ]
            }
        ]
    },
    shakespeare: {
        heightened: [
            {
                title: "Hamlet's Soliloquy",
                context: "Hamlet contemplating life and death",
                script: `To be, or not to be: that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them?

<span class="stage-direction">(pause)</span>

To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to...

<span class="stage-direction">(considering)</span>

'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause.`,
                tips: [
                    "This is thinking out loud, not reciting",
                    "Each thought leads naturally to the next",
                    "Find the personal connection to mortality",
                    "Don't play 'crazy' - play genuine contemplation"
                ]
            },
            {
                title: "Romeo - O, She Doth Teach",
                context: "Romeo seeing Juliet for the first time",
                script: `O, she doth teach the torches to burn bright!
It seems she hangs upon the cheek of night
Like a rich jewel in an Ethiope's ear;

<span class="stage-direction">(mesmerized)</span>

Beauty too rich for use, for earth too dear!
So shows a snowy dove trooping with crows,
As yonder lady o'er her fellows shows.

<span class="stage-direction">(deciding)</span>

Did my heart love till now? Forswear it, sight!
For I ne'er saw true beauty till this night.`,
                tips: [
                    "This is love at first sight - be genuinely stunned",
                    "The poetry should feel spontaneous",
                    "Let us see his whole world shift",
                    "Young love is overwhelming and absolute"
                ]
            }
        ],
        intense: [
            {
                title: "Lady Macbeth - Unsex Me Here",
                context: "Lady Macbeth calling on dark spirits",
                script: `Come, you spirits
That tend on mortal thoughts, unsex me here,
And fill me from the crown to the toe top-full
Of direst cruelty!

<span class="stage-direction">(building)</span>

Make thick my blood;
Stop up the access and passage to remorse,
That no compunctious visitings of nature
Shake my fell purpose, nor keep peace between
The effect and it!

<span class="stage-direction">(commanding)</span>

Come to my woman's breasts,
And take my milk for gall, you murdering ministers!

<span class="stage-direction">(invoking)</span>

Come, thick night,
And pall thee in the dunnest smoke of hell,
That my keen knife see not the wound it makes,
Nor heaven peep through the blanket of the dark,
To cry 'Hold, hold!'`,
                tips: [
                    "This is an invocation - commit fully",
                    "Find the primal desire for power",
                    "Don't play evil - play ambition",
                    "The verse structure drives the build"
                ]
            }
        ]
    }
};

// Script panel state
let scriptState = {
    currentCategory: 'monologues',
    currentIntensity: 'neutral',
    currentIndex: 0,
    isVisible: true
};

// Get scripts for current category/intensity
function getCurrentScripts() {
    const category = SAMPLE_SCRIPTS[scriptState.currentCategory];
    if (!category) return [];
    const scripts = category[scriptState.currentIntensity];
    return scripts || [];
}

// Load and display current script
function loadScript() {
    const scripts = getCurrentScripts();
    if (scripts.length === 0) {
        document.getElementById('scriptTitle').textContent = 'No scripts available';
        document.getElementById('scriptContext').textContent = 'Try selecting a different category or intensity';
        document.getElementById('scriptContent').innerHTML = '';
        document.getElementById('scriptTips').innerHTML = '';
        document.getElementById('scriptCounter').textContent = '0 / 0';
        return;
    }
    
    // Ensure index is valid
    if (scriptState.currentIndex >= scripts.length) {
        scriptState.currentIndex = 0;
    }
    
    const script = scripts[scriptState.currentIndex];
    
    // Update UI
    document.getElementById('scriptTitle').textContent = script.title;
    document.getElementById('scriptContext').textContent = script.context;
    document.getElementById('scriptContent').innerHTML = script.script.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
    document.getElementById('scriptTips').innerHTML = script.tips.map(tip => `<li>${tip}</li>`).join('');
    document.getElementById('scriptCounter').textContent = `${scriptState.currentIndex + 1} / ${scripts.length}`;
}

// Navigate scripts
function nextScript() {
    const scripts = getCurrentScripts();
    if (scripts.length === 0) return;
    scriptState.currentIndex = (scriptState.currentIndex + 1) % scripts.length;
    loadScript();
}

function prevScript() {
    const scripts = getCurrentScripts();
    if (scripts.length === 0) return;
    scriptState.currentIndex = (scriptState.currentIndex - 1 + scripts.length) % scripts.length;
    loadScript();
}

// Toggle script panel visibility
function toggleScriptPanel() {
    const panel = document.getElementById('scriptPanel');
    const showBtn = document.getElementById('showScriptBtn');
    
    scriptState.isVisible = !scriptState.isVisible;
    
    if (scriptState.isVisible) {
        panel.classList.remove('hidden');
        showBtn.classList.add('hidden');
        document.body.classList.add('script-panel-open');
    } else {
        panel.classList.add('hidden');
        showBtn.classList.remove('hidden');
        document.body.classList.remove('script-panel-open');
    }
}

// Initialize script panel
function initScriptPanel() {
    const panel = document.getElementById('scriptPanel');
    if (!panel) return;
    
    // Category selector
    document.getElementById('scriptCategory')?.addEventListener('change', (e) => {
        scriptState.currentCategory = e.target.value;
        scriptState.currentIndex = 0;
        loadScript();
    });
    
    // Intensity selector
    document.getElementById('scriptIntensity')?.addEventListener('change', (e) => {
        scriptState.currentIntensity = e.target.value;
        scriptState.currentIndex = 0;
        loadScript();
    });
    
    // Navigation buttons
    document.getElementById('prevScriptBtn')?.addEventListener('click', prevScript);
    document.getElementById('nextScriptBtn')?.addEventListener('click', nextScript);
    
    // Toggle button
    document.getElementById('toggleScriptBtn')?.addEventListener('click', toggleScriptPanel);
    document.getElementById('showScriptBtn')?.addEventListener('click', toggleScriptPanel);
    
    // Load initial script
    loadScript();
    
    // Start with panel visible
    document.body.classList.add('script-panel-open');
    
    console.log('📜 Script panel initialized');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initScriptPanel);
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(initScriptPanel, 100);
}

// ============================================================================
// DIRECTOR'S ANALYSIS SYSTEM
// ============================================================================

/**
 * AI Director's Analysis Philosophy:
 * "Every word, pause, and breath tells a story. A great director sees not just 
 * what's there, but what could be. This analysis reveals the hidden patterns 
 * in your performance that separate good from extraordinary."
 */

// Director's Analysis state
const directorAnalysis = {
    fillerWords: {},
    fillerTimeline: [],
    grammarIssues: [],
    emotionalHistory: [],
    energyHistory: [],
    pauseHistory: [],
    scriptWords: [],
    transcriptWords: [],
    comparisonResults: null
};

// Common filler words to detect
const FILLER_WORDS = [
    'um', 'uh', 'er', 'ah', 'like', 'you know', 'basically', 'actually',
    'literally', 'honestly', 'so', 'well', 'right', 'okay', 'i mean',
    'sort of', 'kind of', 'you see', 'mmm', 'hmm'
];

// Initialize analysis panel
function initAnalysisPanel() {
    // Tab switching
    document.querySelectorAll('.analysis-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active from all tabs and contents
            document.querySelectorAll('.analysis-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.analysis-content').forEach(c => c.classList.remove('active'));
            
            // Add active to clicked tab and corresponding content
            tab.classList.add('active');
            const contentId = 'analysis' + tab.dataset.analysis.charAt(0).toUpperCase() + tab.dataset.analysis.slice(1);
            document.getElementById(contentId)?.classList.add('active');
        });
    });
    
    console.log('🎬 Director\'s Analysis Panel initialized');
}

// Reset analysis for new session
function resetAnalysis() {
    directorAnalysis.fillerWords = {};
    directorAnalysis.fillerTimeline = [];
    directorAnalysis.grammarIssues = [];
    directorAnalysis.emotionalHistory = [];
    directorAnalysis.energyHistory = [];
    directorAnalysis.pauseHistory = [];
    directorAnalysis.scriptWords = [];
    directorAnalysis.transcriptWords = [];
    directorAnalysis.comparisonResults = null;
    
    // Reset UI
    updateFillerWordsUI();
    updateGrammarUI();
    updateScriptComparison();
    updateProfessionalMetrics();
}

// Analyze transcript for fillers, grammar, and compare to script
function analyzeTranscript(transcript, timestamp) {
    if (!transcript || transcript.trim().length === 0) return;
    
    const words = transcript.toLowerCase().split(/\s+/);
    directorAnalysis.transcriptWords = words;
    
    // Detect filler words
    detectFillerWords(transcript, timestamp);
    
    // Analyze grammar
    analyzeGrammar(transcript);
    
    // Compare to script if available
    compareToScript();
    
    // Update all UI
    updateFillerWordsUI();
    updateGrammarUI();
    updateScriptComparison();
}

// Detect filler words in text
function detectFillerWords(text, timestamp) {
    const lowerText = text.toLowerCase();
    const sessionTime = (Date.now() - state.sessionStartTime) / 1000;
    
    FILLER_WORDS.forEach(filler => {
        const regex = new RegExp(`\\b${filler}\\b`, 'gi');
        const matches = lowerText.match(regex);
        
        if (matches) {
            // Count occurrences
            if (!directorAnalysis.fillerWords[filler]) {
                directorAnalysis.fillerWords[filler] = 0;
            }
            
            // Check for new occurrences
            const currentCount = directorAnalysis.fillerWords[filler];
            const newCount = matches.length;
            
            // Add new ones to timeline
            for (let i = currentCount; i < newCount; i++) {
                directorAnalysis.fillerTimeline.push({
                    word: filler,
                    time: sessionTime,
                    timestamp: Date.now()
                });
            }
            
            directorAnalysis.fillerWords[filler] = newCount;
        }
    });
}

// Update filler words UI
function updateFillerWordsUI() {
    // Total count
    const totalFillers = Object.values(directorAnalysis.fillerWords).reduce((a, b) => a + b, 0);
    const totalWords = directorAnalysis.transcriptWords.length || 1;
    const fillerRate = ((totalFillers / totalWords) * 100).toFixed(1);
    
    document.getElementById('totalFillerCount').textContent = totalFillers;
    document.getElementById('fillerRate').textContent = fillerRate + '%';
    
    // Trend (compare first half to second half of session)
    const midpoint = directorAnalysis.fillerTimeline.length / 2;
    const firstHalf = directorAnalysis.fillerTimeline.filter((_, i) => i < midpoint).length;
    const secondHalf = directorAnalysis.fillerTimeline.filter((_, i) => i >= midpoint).length;
    
    let trend = '→';
    if (secondHalf > firstHalf * 1.2) trend = '↗ Increasing';
    else if (secondHalf < firstHalf * 0.8) trend = '↘ Improving';
    else trend = '→ Stable';
    
    document.getElementById('fillerTrend').textContent = trend;
    
    // Filler breakdown bars
    const fillerBars = document.getElementById('fillerBars');
    if (fillerBars) {
        const sortedFillers = Object.entries(directorAnalysis.fillerWords)
            .filter(([_, count]) => count > 0)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);
        
        const maxCount = sortedFillers.length > 0 ? sortedFillers[0][1] : 1;
        
        fillerBars.innerHTML = sortedFillers.map(([word, count]) => `
            <div class="filler-bar-item">
                <span class="filler-word">"${word}"</span>
                <div class="filler-bar">
                    <div class="filler-bar-fill" style="width: ${(count / maxCount) * 100}%"></div>
                </div>
                <span class="filler-bar-count">${count}</span>
            </div>
        `).join('') || '<p class="no-issues">No filler words detected yet</p>';
    }
    
    // Timeline
    const timeline = document.getElementById('fillerTimeline');
    if (timeline && directorAnalysis.fillerTimeline.length > 0) {
        timeline.innerHTML = directorAnalysis.fillerTimeline.slice(-20).map(item => `
            <span class="timeline-item">
                <span class="timeline-word">${item.word}</span>
                <span class="timeline-time">${formatTime(item.time)}</span>
            </span>
        `).join('');
    }
    
    // Draw chart
    drawFillersChart();
}

// Format time as MM:SS
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Draw fillers chart
function drawFillersChart() {
    const canvas = document.getElementById('fillersChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.fillRect(0, 0, width, height);
    
    // Group fillers by 30-second intervals
    const sessionDuration = (Date.now() - (state.sessionStartTime || Date.now())) / 1000;
    const intervalSeconds = 30;
    const intervals = Math.max(1, Math.ceil(sessionDuration / intervalSeconds));
    
    const buckets = new Array(intervals).fill(0);
    directorAnalysis.fillerTimeline.forEach(item => {
        const bucketIndex = Math.min(Math.floor(item.time / intervalSeconds), intervals - 1);
        buckets[bucketIndex]++;
    });
    
    const maxBucket = Math.max(...buckets, 1);
    const barWidth = (width - 40) / intervals;
    
    // Draw bars
    buckets.forEach((count, i) => {
        const barHeight = (count / maxBucket) * (height - 30);
        const x = 20 + i * barWidth;
        const y = height - 20 - barHeight;
        
        // Gradient based on count
        const gradient = ctx.createLinearGradient(x, y + barHeight, x, y);
        gradient.addColorStop(0, '#f87171');
        gradient.addColorStop(1, '#fbbf24');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth - 2, barHeight);
        
        // Count label
        if (count > 0) {
            ctx.fillStyle = '#fff';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(count.toString(), x + barWidth / 2, y - 5);
        }
    });
    
    // Draw axis
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.beginPath();
    ctx.moveTo(20, height - 20);
    ctx.lineTo(width - 20, height - 20);
    ctx.stroke();
    
    // Time labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i <= intervals; i += Math.max(1, Math.floor(intervals / 4))) {
        const x = 20 + i * barWidth;
        ctx.fillText(formatTime(i * intervalSeconds), x, height - 5);
    }
}

// Analyze grammar in transcript
function analyzeGrammar(transcript) {
    const issues = [];
    
    // Common grammar patterns to check
    const grammarChecks = [
        { pattern: /\bi\b(?!\s*(am|was|will|would|have|had|'m|'ll|'d|'ve))/gi, issue: 'Lowercase "I"', suggestion: 'Capitalize "I"', type: 'error' },
        { pattern: /\b(gonna|wanna|gotta)\b/gi, issue: 'Informal contraction', suggestion: 'Use formal: going to/want to/got to', type: 'warning' },
        { pattern: /\b(ain't)\b/gi, issue: 'Non-standard contraction', suggestion: 'Consider: am not/is not/are not', type: 'warning' },
        { pattern: /\b(should of|could of|would of)\b/gi, issue: 'Common mistake', suggestion: 'Use: should have/could have/would have', type: 'error' },
        { pattern: /\b(irregardless)\b/gi, issue: 'Non-standard word', suggestion: 'Use: regardless', type: 'error' },
        { pattern: /\b(supposably)\b/gi, issue: 'Mispronunciation', suggestion: 'Use: supposedly', type: 'error' },
        { pattern: /\b(alot)\b/gi, issue: 'Spelling error', suggestion: 'Use: a lot (two words)', type: 'error' },
        { pattern: /\b(could care less)\b/gi, issue: 'Idiom error', suggestion: 'Use: couldn\'t care less', type: 'warning' },
        { pattern: /\.{4,}/g, issue: 'Excessive punctuation', suggestion: 'Use standard ellipsis (...)', type: 'warning' },
        { pattern: /\b(me and \w+)\b/gi, issue: 'Grammar order', suggestion: 'Consider: [person] and I/me', type: 'warning' }
    ];
    
    grammarChecks.forEach(check => {
        const matches = transcript.match(check.pattern);
        if (matches) {
            matches.forEach(match => {
                issues.push({
                    type: check.type,
                    text: check.issue + `: "${match}"`,
                    suggestion: check.suggestion
                });
            });
        }
    });
    
    directorAnalysis.grammarIssues = issues;
    
    // Calculate scores
    const totalWords = directorAnalysis.transcriptWords.length || 1;
    const errorCount = issues.filter(i => i.type === 'error').length;
    const warningCount = issues.filter(i => i.type === 'warning').length;
    
    // Grammar score (100 - errors*5 - warnings*2)
    const grammarScore = Math.max(0, Math.min(100, 100 - errorCount * 5 - warningCount * 2));
    
    return {
        score: grammarScore,
        pronunciation: 85 + Math.random() * 15, // Simulated
        structure: 80 + Math.random() * 20,
        wordChoice: 75 + Math.random() * 25,
        tense: 90 + Math.random() * 10
    };
}

// Update grammar UI
function updateGrammarUI() {
    const scores = analyzeGrammar(state.fullTranscript || '');
    
    // Update score circle
    const scoreValue = document.getElementById('grammarScoreValue');
    const scoreCircle = document.getElementById('grammarScoreCircle');
    if (scoreValue) scoreValue.textContent = Math.round(scores.score);
    if (scoreCircle) {
        scoreCircle.style.background = `conic-gradient(var(--accent-gold) ${scores.score * 3.6}deg, var(--bg-elevated) 0deg)`;
    }
    
    // Update breakdown bars
    const setBar = (id, value) => {
        const bar = document.getElementById(id + 'Bar');
        const val = document.getElementById(id + 'Value');
        if (bar) bar.style.width = value + '%';
        if (val) val.textContent = Math.round(value);
    };
    
    setBar('pronunciation', scores.pronunciation);
    setBar('structure', scores.structure);
    setBar('wordChoice', scores.wordChoice);
    setBar('tense', scores.tense);
    
    // Update issues list
    const issuesList = document.getElementById('issuesList');
    if (issuesList) {
        if (directorAnalysis.grammarIssues.length === 0) {
            issuesList.innerHTML = '<p class="no-issues">No issues detected yet. Start speaking to analyze...</p>';
        } else {
            issuesList.innerHTML = directorAnalysis.grammarIssues.slice(0, 5).map(issue => `
                <div class="issue-item ${issue.type}">
                    <span class="issue-icon">${issue.type === 'error' ? '❌' : '⚠️'}</span>
                    <div>
                        <div class="issue-text">${issue.text}</div>
                        <div class="issue-suggestion">💡 ${issue.suggestion}</div>
                    </div>
                </div>
            `).join('');
        }
    }
}

// Compare transcript to selected script
function compareToScript() {
    // Get current script from the script panel
    const scripts = getCurrentScripts();
    if (!scripts || scripts.length === 0) return;
    
    const currentScript = scripts[scriptState.currentIndex];
    if (!currentScript) return;
    
    // Extract plain text from script (remove HTML tags and stage directions)
    const scriptText = currentScript.script
        .replace(/<[^>]*>/g, '') // Remove HTML tags
        .replace(/\([^)]*\)/g, '') // Remove stage directions
        .replace(/\s+/g, ' ') // Normalize whitespace
        .trim()
        .toLowerCase();
    
    const transcriptText = (state.fullTranscript || '')
        .toLowerCase()
        .replace(/[^\w\s]/g, '') // Remove punctuation
        .replace(/\s+/g, ' ')
        .trim();
    
    if (!transcriptText) {
        directorAnalysis.comparisonResults = null;
        return;
    }
    
    const scriptWords = scriptText.split(/\s+/).filter(w => w.length > 0);
    const transcriptWords = transcriptText.split(/\s+/).filter(w => w.length > 0);
    
    // Simple diff algorithm
    const result = {
        words: [],
        correct: 0,
        missed: 0,
        added: 0,
        wrong: 0
    };
    
    let scriptIndex = 0;
    let transcriptIndex = 0;
    
    while (scriptIndex < scriptWords.length || transcriptIndex < transcriptWords.length) {
        const scriptWord = scriptWords[scriptIndex];
        const transcriptWord = transcriptWords[transcriptIndex];
        
        if (scriptWord === transcriptWord) {
            // Correct match
            result.words.push({ word: transcriptWord, type: 'correct' });
            result.correct++;
            scriptIndex++;
            transcriptIndex++;
        } else if (!transcriptWord) {
            // Missed word (script has more)
            result.words.push({ word: scriptWord, type: 'missed' });
            result.missed++;
            scriptIndex++;
        } else if (!scriptWord) {
            // Added word (transcript has more)
            result.words.push({ word: transcriptWord, type: 'added' });
            result.added++;
            transcriptIndex++;
        } else {
            // Look ahead to find matches
            let foundInScript = false;
            let foundInTranscript = false;
            
            // Check if transcript word appears later in script
            for (let i = scriptIndex + 1; i < Math.min(scriptIndex + 5, scriptWords.length); i++) {
                if (scriptWords[i] === transcriptWord) {
                    foundInScript = true;
                    break;
                }
            }
            
            // Check if script word appears later in transcript
            for (let i = transcriptIndex + 1; i < Math.min(transcriptIndex + 5, transcriptWords.length); i++) {
                if (transcriptWords[i] === scriptWord) {
                    foundInTranscript = true;
                    break;
                }
            }
            
            if (foundInScript && !foundInTranscript) {
                // Script word was missed
                result.words.push({ word: scriptWord, type: 'missed' });
                result.missed++;
                scriptIndex++;
            } else if (foundInTranscript && !foundInScript) {
                // Extra word added
                result.words.push({ word: transcriptWord, type: 'added' });
                result.added++;
                transcriptIndex++;
            } else {
                // Wrong word substitution
                result.words.push({ word: `${scriptWord}→${transcriptWord}`, type: 'wrong' });
                result.wrong++;
                scriptIndex++;
                transcriptIndex++;
            }
        }
    }
    
    directorAnalysis.comparisonResults = result;
}

// Update script comparison UI
function updateScriptComparison() {
    const result = directorAnalysis.comparisonResults;
    
    if (!result) {
        document.getElementById('accuracyScore').textContent = '--%';
        document.getElementById('missedWords').textContent = '--';
        document.getElementById('addedWords').textContent = '--';
        document.getElementById('wordOrderScore').textContent = '--%';
        document.getElementById('scriptComparisonText').innerHTML = 
            '<p class="comparison-placeholder">Select a script from the panel and start rehearsing to see comparison...</p>';
        return;
    }
    
    const total = result.correct + result.missed + result.wrong;
    const accuracy = total > 0 ? Math.round((result.correct / total) * 100) : 0;
    const wordOrder = Math.max(0, 100 - result.wrong * 10);
    
    document.getElementById('accuracyScore').textContent = accuracy + '%';
    document.getElementById('missedWords').textContent = result.missed;
    document.getElementById('addedWords').textContent = result.added;
    document.getElementById('wordOrderScore').textContent = wordOrder + '%';
    
    // Render comparison view
    const comparisonText = document.getElementById('scriptComparisonText');
    if (comparisonText) {
        comparisonText.innerHTML = result.words.map(item => {
            const className = 'word-' + item.type;
            return `<span class="${className}">${item.word}</span>`;
        }).join(' ');
    }
}

// Update professional metrics
function updateProfessionalMetrics() {
    // These are updated from the real-time metrics
    // Eye contact time
    const eyeContactEl = document.getElementById('eyeContactTime');
    if (eyeContactEl && state.eyeContactPercentage !== undefined) {
        eyeContactEl.textContent = Math.round(state.eyeContactPercentage || 50) + '%';
    }
    
    // Framing score (from face position)
    const framingEl = document.getElementById('framingScore');
    if (framingEl) {
        framingEl.textContent = state.framingScore || 'Good';
    }
    
    // Movement/Stillness
    const movementEl = document.getElementById('movementScore');
    if (movementEl) {
        movementEl.textContent = state.stillnessScore || 'Stable';
    }
    
    // Pause analysis
    const pauseCount = directorAnalysis.pauseHistory.length;
    document.getElementById('totalPauses').textContent = pauseCount;
    
    const avgPause = pauseCount > 0 
        ? (directorAnalysis.pauseHistory.reduce((a, b) => a + b, 0) / pauseCount).toFixed(1)
        : '--';
    document.getElementById('avgPauseDuration').textContent = avgPause + 's';
    
    const dramaticPauses = directorAnalysis.pauseHistory.filter(p => p > 1.5).length;
    document.getElementById('dramaticPauses').textContent = dramaticPauses;
    
    // Take quality
    const qualityFill = document.getElementById('takeQualityFill');
    const qualityVerdict = document.getElementById('takeQualityVerdict');
    
    // Calculate overall quality based on various factors
    const fillerPenalty = Math.min(30, Object.values(directorAnalysis.fillerWords).reduce((a, b) => a + b, 0) * 3);
    const grammarPenalty = directorAnalysis.grammarIssues.length * 5;
    const accuracyBonus = directorAnalysis.comparisonResults 
        ? (directorAnalysis.comparisonResults.correct / Math.max(1, directorAnalysis.comparisonResults.correct + directorAnalysis.comparisonResults.missed)) * 30
        : 15;
    
    const quality = Math.max(0, Math.min(100, 70 - fillerPenalty - grammarPenalty + accuracyBonus));
    
    if (qualityFill) qualityFill.style.width = quality + '%';
    if (qualityVerdict) {
        if (quality >= 80) qualityVerdict.textContent = '🌟 Excellent take! Ready for review.';
        else if (quality >= 60) qualityVerdict.textContent = '✓ Good take. Minor improvements possible.';
        else if (quality >= 40) qualityVerdict.textContent = '⚡ Decent effort. Consider another take.';
        else qualityVerdict.textContent = '🔄 Needs work. Focus on the basics.';
    }
    
    // Draw emotional arc
    drawEmotionalArc();
    
    // Draw energy graph
    drawEnergyGraph();
}

// Draw emotional arc visualization
function drawEmotionalArc() {
    const canvas = document.getElementById('emotionalArcCanvas');
    if (!canvas || directorAnalysis.emotionalHistory.length < 2) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    // Draw arc line
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const emotions = directorAnalysis.emotionalHistory;
    const step = width / Math.max(1, emotions.length - 1);
    
    emotions.forEach((emotion, i) => {
        const x = i * step;
        const y = height - (emotion.intensity * height);
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    
    // Fill area under curve
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fillStyle = 'rgba(212, 175, 55, 0.1)';
    ctx.fill();
}

// Draw energy graph
function drawEnergyGraph() {
    const canvas = document.getElementById('energyGraphCanvas');
    if (!canvas || directorAnalysis.energyHistory.length < 2) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    ctx.clearRect(0, 0, width, height);
    
    // Draw energy line
    ctx.strokeStyle = 'rgba(74, 222, 128, 0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const energies = directorAnalysis.energyHistory;
    const step = width / Math.max(1, energies.length - 1);
    const maxEnergy = Math.max(...energies, 1);
    
    energies.forEach((energy, i) => {
        const x = i * step;
        const y = height - (energy / maxEnergy * height * 0.9) - 5;
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    
    // Update stats
    const avgEnergy = (energies.reduce((a, b) => a + b, 0) / energies.length).toFixed(0);
    const peakEnergy = Math.max(...energies).toFixed(0);
    const lowEnergy = Math.min(...energies).toFixed(0);
    
    document.getElementById('avgEnergy').textContent = avgEnergy + 'dB';
    document.getElementById('peakEnergy').textContent = peakEnergy + 'dB';
    document.getElementById('lowEnergy').textContent = lowEnergy + 'dB';
}

// Track emotional changes
function trackEmotionalState(emotion, intensity) {
    directorAnalysis.emotionalHistory.push({
        emotion: emotion,
        intensity: intensity,
        time: Date.now()
    });
    
    // Keep only last 50 readings
    if (directorAnalysis.emotionalHistory.length > 50) {
        directorAnalysis.emotionalHistory.shift();
    }
}

// Track energy levels
function trackEnergyLevel(db) {
    directorAnalysis.energyHistory.push(db);
    
    // Keep only last 50 readings
    if (directorAnalysis.energyHistory.length > 50) {
        directorAnalysis.energyHistory.shift();
    }
}

// Track pauses
function trackPause(duration) {
    if (duration > 0.3) { // Only track pauses > 300ms
        directorAnalysis.pauseHistory.push(duration);
    }
}

// Hook into existing metrics update to track analysis
const originalUpdateMetrics3 = updateMetrics;
updateMetrics = function(data) {
    originalUpdateMetrics3(data);
    
    // Track energy
    if (data.volume_db !== undefined) {
        trackEnergyLevel(Math.max(0, data.volume_db + 60)); // Normalize to positive
    }
    
    // Track emotions
    if (data.voice_emotion) {
        const emotionIntensity = {
            'neutral': 0.3,
            'happy': 0.7,
            'sad': 0.5,
            'angry': 0.9,
            'fearful': 0.6,
            'surprised': 0.8
        };
        trackEmotionalState(data.voice_emotion, emotionIntensity[data.voice_emotion] || 0.5);
    }
    
    // Update professional metrics
    updateProfessionalMetrics();
};

// Hook into transcript updates for analysis
const originalUpdateTranscript = typeof updateTranscript === 'function' ? updateTranscript : null;
function updateTranscriptWithAnalysis(text) {
    if (originalUpdateTranscript) originalUpdateTranscript(text);
    
    // Run analysis
    analyzeTranscript(text, Date.now());
}

// Override transcript handling
if (typeof handleTranscript === 'function') {
    const originalHandleTranscript = handleTranscript;
    handleTranscript = function(data) {
        originalHandleTranscript(data);
        if (data.text) {
            analyzeTranscript(state.fullTranscript, Date.now());
        }
    };
}

// Initialize analysis panel on load
document.addEventListener('DOMContentLoaded', initAnalysisPanel);
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(initAnalysisPanel, 100);
}

// Reset analysis when session starts
const originalStartSession2 = startSession;
startSession = async function() {
    resetAnalysis();
    await originalStartSession2.apply(this, arguments);
};

// ============================================================================
// COACH SUBSCRIPTION & NOTIFY SYSTEM
// ============================================================================

let currentNotifyCoach = null;

// Open notify modal
function openNotifyModal(coachName, icon) {
    currentNotifyCoach = coachName;
    const modal = document.getElementById('notifyModal');
    const iconEl = document.getElementById('notifyModalIcon');
    const titleEl = document.getElementById('notifyModalTitle');
    const descEl = document.getElementById('notifyModalDesc');
    
    if (!modal) return;
    
    // Set content based on coach
    const coachNames = {
        'interview': 'Interview Prep Coach',
        'family': 'Family & Marriage Coach',
        'life': 'Life Coach',
        'presentation': 'Presentation Coach',
        'yoga': 'Yoga Coach',
        'martial': 'Martial Arts Coach',
        'gym': 'Gym Coach',
        'swim': 'Swim Coach',
        'dance': 'Dance Coach',
        'music': 'Music Coach'
    };
    
    const coachIcons = {
        'interview': '💼',
        'family': '👨‍👩‍👧‍👦',
        'life': '🌟',
        'presentation': '📊',
        'yoga': '🧘',
        'martial': '🥋',
        'gym': '🏋️',
        'swim': '🏊',
        'dance': '💃',
        'music': '🎸'
    };
    
    iconEl.textContent = coachIcons[coachName] || '🔔';
    titleEl.textContent = `Get Notified: ${coachNames[coachName] || coachName}`;
    descEl.textContent = `Be the first to know when the ${coachNames[coachName] || coachName} launches!`;
    
    modal.classList.remove('hidden');
}

// Close notify modal
function closeNotifyModal() {
    const modal = document.getElementById('notifyModal');
    if (modal) modal.classList.add('hidden');
    currentNotifyCoach = null;
}

// Submit notification request
function submitNotify() {
    const emailInput = document.getElementById('notifyEmail');
    const email = emailInput?.value?.trim();
    
    if (!email || !email.includes('@')) {
        alert('Please enter a valid email address');
        return;
    }
    
    // Store notification request
    const notifications = JSON.parse(localStorage.getItem('coachNotifications') || '[]');
    notifications.push({
        coach: currentNotifyCoach,
        email: email,
        date: new Date().toISOString()
    });
    localStorage.setItem('coachNotifications', JSON.stringify(notifications));
    
    // Show success
    const modal = document.getElementById('notifyModal');
    if (modal) {
        modal.querySelector('.notify-modal-content').innerHTML = `
            <div class="notify-icon">✅</div>
            <h3>You're on the list!</h3>
            <p>We'll email you at <strong>${email}</strong> when this coach launches.</p>
            <button class="btn btn-primary" onclick="closeNotifyModal()">Got it!</button>
        `;
    }
    
    console.log(`📧 Notification registered: ${email} for ${currentNotifyCoach}`);
}

// Initialize coach buttons
function initCoachButtons() {
    document.querySelectorAll('.btn-subscribe').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const coachCard = e.target.closest('.coach-card');
            const coachName = coachCard?.dataset?.coach;
            if (coachName) {
                openNotifyModal(coachName);
            }
        });
    });
    
    // Close modal on outside click
    document.getElementById('notifyModal')?.addEventListener('click', (e) => {
        if (e.target.id === 'notifyModal') {
            closeNotifyModal();
        }
    });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initCoachButtons);
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setTimeout(initCoachButtons, 100);
}
