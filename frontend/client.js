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
    // WebSocket
    WS_URL: `ws://${window.location.host}/ws`,
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
    const bufferSize = Math.ceil(CONFIG.AUDIO_SAMPLE_RATE * CONFIG.AUDIO_CHUNK_MS / 1000);
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
// TAB NAVIGATION
// ============================================================================

const tabs = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const targetTab = tab.dataset.tab;
        
        // Update active tab button
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
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
        
        console.log(`📑 Switched to tab: ${targetTab}`);
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
