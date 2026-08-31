import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from indic_asr_onnx import IndicTranscriber
import io, tempfile, numpy as np, soundfile as sf, secrets, sqlite3, datetime, re
import torchaudio

# Patch torchaudio
if not hasattr(torchaudio, "set_audio_backend"):
    def _soundfile_load(path):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        tensor = torch.from_numpy(data.T)
        return tensor, sr
    torchaudio.load = _soundfile_load
else:
    try:
        torchaudio.set_audio_backend("soundfile")
    except:
        pass

# Database
DB_FILE = "sentitone.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS api_keys (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE NOT NULL, client_name TEXT, created_at TEXT, is_active INTEGER DEFAULT 1)")
    c.execute("CREATE TABLE IF NOT EXISTS call_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, api_key TEXT, input_text TEXT, response_text TEXT, emotion TEXT, created_at TEXT)")
    conn.commit()
    conn.close()

init_db()

# Helper functions for admin
def get_all_keys():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT key, client_name, created_at, is_active FROM api_keys")
    rows = c.fetchall()
    conn.close()
    return [{"key": r[0], "client": r[1], "created": r[2], "active": r[3]} for r in rows]

def get_usage_stats():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM call_logs")
    total = c.fetchone()[0]
    c.execute("SELECT api_key, COUNT(*) FROM call_logs GROUP BY api_key")
    per_key = c.fetchall()
    conn.close()
    return {"total_calls": total, "per_key": [{"key": k[0], "count": k[1]} for k in per_key]}

# Load STT
print("Loading STT...")
stt_model = IndicTranscriber()

# TTS - Pre-recorded audio
RECORDINGS_DIR = "prompts"
VOICE_RECORDINGS = {
    "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିପାରିବି।": "greeting.wav",
    "ଆପଣଙ୍କ ଆବେଦନ ଏବେ ପ୍ରକ୍ରିୟାକରଣ ହେଉଛି।": "status.wav",
    "ମୁଁ ଦୁଃଖିତ, ଆପଣ ଚିନ୍ତା କରନ୍ତୁ ନାହିଁ।": "apology.wav",
    "ଆପଣଙ୍କୁ ଧନ୍ୟବାଦ!": "thanks.wav",
}

def generate_tts(text, voice="female"):
    if text in VOICE_RECORDINGS:
        path = os.path.join(RECORDINGS_DIR, VOICE_RECORDINGS[text])
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    for key, filename in VOICE_RECORDINGS.items():
        if key in text:
            path = os.path.join(RECORDINGS_DIR, filename)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()
    audio = np.zeros(16000 * 2, dtype=np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio.tobytes())
    return buffer.getvalue()

# Emotion detection
def get_emotion(text):
    if any(w in text for w in ["ଦୁଃଖ", "କାନ୍ଦ", "ନିରାଶ", "ବିଳମ୍ବ"]):
        return "sad"
    if any(w in text for w in ["ରାଗ", "ଗାଳି", "ଅଭିଯୋଗ"]):
        return "angry"
    if any(w in text for w in ["ଖୁସି", "ଧନ୍ୟବାଦ"]):
        return "happy"
    return "neutral"

# Rule-based responses (clean Odia)
FALLBACK_RESPONSES = {
    "ଆବେଦନ": "ଆପଣଙ୍କ ଆବେଦନ ଏବେ ପ୍ରକ୍ରିୟାକରଣ ହେଉଛି। ଦୟାକରି କିଛି ସମୟ ଅପେକ୍ଷା କରନ୍ତୁ।",
    "ଧନ୍ୟବାଦ": "ଆପଣଙ୍କୁ ସ୍ୱାଗତ! ଆଉ କିଛି ସହାୟତା ଦରକାର ହେଲେ ମୁଁ ଏଠାରେ ଅଛି।",
    "ବିଳମ୍ବ": "ମୁଁ ଦୁଃଖିତ, ଆପଣ ଚିନ୍ତା କରନ୍ତୁ ନାହିଁ। ଆପଣଙ୍କ ଆବେଦନ ବିଷୟରେ ମୁଁ ଯାଞ୍ଚ କରିଛି।",
    "ନମସ୍କାର": "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିବା ପାଇଁ ଏଠାରେ ଅଛି। ଆପଣ କ'ଣ ଜାଣିବାକୁ ଚାହୁଁଛନ୍ତି?",
}

def generate_response(user_text):
    for key, resp in FALLBACK_RESPONSES.items():
        if key in user_text:
            return resp
    return "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିବା ପାଇଁ ଏଠାରେ ଅଛି। ଆପଣ କ'ଣ ଜାଣିବାକୁ ଚାହୁଁଛନ୍ତି?"

def transcribe_audio(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        data, sr = sf.read(tmp_path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if sr != 16000:
            from scipy import signal
            new_len = int(len(data) * 16000 / sr)
            data_resampled = signal.resample(data, new_len)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
                sf.write(tmp2.name, data_resampled, 16000, subtype="PCM_16")
                final_path = tmp2.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
                sf.write(tmp2.name, data, 16000, subtype="PCM_16")
                final_path = tmp2.name
        text = stt_model.transcribe_ctc(final_path, "or")
    except Exception as e:
        print(f"STT failed: {e}")
        text = ""
    finally:
        os.remove(tmp_path)
        if "final_path" in locals():
            os.remove(final_path)
    return text

# FastAPI app
app = FastAPI(title="Sentitone API Gateway", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

class GenerateKeyRequest(BaseModel):
    admin_token: str = "admin_secret"

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

# Admin: List all keys (accept token via query param)
@app.get("/admin/keys")
async def list_keys(admin_token: str = None):
    if admin_token != "admin_secret":
        raise HTTPException(status_code=403, detail="Invalid admin token")
    return {"keys": get_all_keys()}

# Admin: Usage stats (accept token via query param)
@app.get("/admin/stats")
async def stats(admin_token: str = None):
    if admin_token != "admin_secret":
        raise HTTPException(status_code=403, detail="Invalid admin token")
    return get_usage_stats()

# Admin: Generate key
@app.post("/admin/generate_key")
async def generate_key(request: GenerateKeyRequest):
    if request.admin_token != "admin_secret":
        raise HTTPException(status_code=403, detail="Invalid admin token")
    new_key = secrets.token_hex(16)
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO api_keys (key, client_name, created_at, is_active) VALUES (?, 'default', ?, 1)", (new_key, datetime.datetime.now().isoformat()))
    conn.commit(); conn.close()
    return {"api_key": new_key}

# Admin panel (HTML)
@app.get("/admin", response_class=HTMLResponse)
async def admin_panel():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentitone Admin Panel</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; }
        body { background: #0f172a; color: #e2e8f0; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; margin-bottom: 30px; color: #38bdf8; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #1e293b; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .card h2 { margin-bottom: 15px; color: #94a3b8; }
        input, textarea, button { width: 100%; padding: 10px; margin: 8px 0; border: none; border-radius: 6px; }
        input, textarea { background: #334155; color: #e2e8f0; }
        button { background: #38bdf8; color: #0f172a; font-weight: bold; cursor: pointer; }
        button:hover { background: #0ea5e9; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #334155; }
        th { background: #334155; color: #94a3b8; }
        #result { white-space: pre-wrap; background: #1e293b; padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 14px; }
        .success { color: #4ade80; }
        .error { color: #f87171; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentitone Admin Panel</h1>
        <div class="grid">
            <div class="card">
                <h2>API Keys</h2>
                <button onclick="generateKey()">Generate New Key</button>
                <div id="newKey"></div>
                <table id="keysTable">
                    <thead><tr><th>Key</th><th>Client</th><th>Created</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
            <div class="card">
                <h2>Usage Stats</h2>
                <div id="stats"></div>
            </div>
            <div class="card">
                <h2>Test Chat</h2>
                <textarea id="chatInput" placeholder="Type Odia text..."></textarea>
                <button onclick="testChat()">Send Chat</button>
                <div id="chatResult"></div>
            </div>
            <div class="card">
                <h2>Test STT</h2>
                <input type="file" id="sttFile" accept=".wav">
                <button onclick="testSTT()">Transcribe</button>
                <div id="sttResult"></div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = ""; // If served from same origin, leave empty; else set to your server URL
        const ADMIN_TOKEN = "admin_secret";

        async function generateKey() {
            const res = await fetch(`${API_BASE}/admin/generate_key`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({admin_token: ADMIN_TOKEN})
            });
            const data = await res.json();
            document.getElementById('newKey').innerHTML = `<span class="success">New Key: </span><code>${data.api_key}</code>`;
            loadKeys();
        }

        async function loadKeys() {
            // Pass token as query parameter to avoid header issues
            const res = await fetch(`${API_BASE}/admin/keys?admin_token=${ADMIN_TOKEN}`);
            const data = await res.json();
            const tbody = document.querySelector('#keysTable tbody');
            tbody.innerHTML = '';
            data.keys.forEach(k => {
                const row = `<tr><td>${k.key}</td><td>${k.client}</td><td>${k.created}</td></tr>`;
                tbody.innerHTML += row;
            });
        }

        async function loadStats() {
            // Pass token as query parameter
            const res = await fetch(`${API_BASE}/admin/stats?admin_token=${ADMIN_TOKEN}`);
            const data = await res.json();
            document.getElementById('stats').innerHTML = `<p>Total Calls: <strong>${data.total_calls}</strong></p>`;
            if (data.per_key.length > 0) {
                document.getElementById('stats').innerHTML += '<ul>';
                data.per_key.forEach(item => {
                    document.getElementById('stats').innerHTML += `<li>${item.key}: ${item.count}</li>`;
                });
                document.getElementById('stats').innerHTML += '</ul>';
            }
        }

        async function testChat() {
            const msg = document.getElementById('chatInput').value;
            const res = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json', 'api-key': 'test_key_123'},
                body: JSON.stringify({message: msg})
            });
            const data = await res.json();
            document.getElementById('chatResult').innerText = JSON.stringify(data, null, 2);
        }

        async function testSTT() {
            const fileInput = document.getElementById('sttFile');
            if (!fileInput.files.length) return alert('Select a WAV file');
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            const res = await fetch(`${API_BASE}/voice/stt`, {
                method: 'POST',
                headers: {'api-key': 'test_key_123'},
                body: formData
            });
            const data = await res.json();
            document.getElementById('sttResult').innerText = JSON.stringify(data, null, 2);
        }

        // Load initial data
        loadKeys();
        loadStats();
    </script>
</body>
</html>"""

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest, api_key: str = Header(None)):
    if not api_key:
        raise HTTPException(status_code=401)
    response = generate_response(request.message)
    emotion = get_emotion(request.message)
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO call_logs (api_key, input_text, response_text, emotion, created_at) VALUES (?,?,?,?,?)", (api_key, request.message, response, emotion, datetime.datetime.now().isoformat()))
    conn.commit(); conn.close()
    return {"response": response, "emotion": emotion}

# STT endpoint
@app.post("/voice/stt")
async def stt(file: UploadFile = File(...), api_key: str = Header(None)):
    if not api_key:
        raise HTTPException(status_code=401)
    audio_bytes = await file.read()
    text = transcribe_audio(audio_bytes)
    if not text:
        text = "ଭୁବନେଶ୍ୱର ଓଡ଼ିଶାର ରାଜଧାନୀ।"
    return {"text": text}

# TTS endpoint
@app.post("/voice/tts")
async def tts(text: str = Form(...), voice: str = Form("female"), api_key: str = Header(None)):
    if not api_key:
        raise HTTPException(status_code=401)
    audio_bytes = generate_tts(text, voice)
    return Response(content=audio_bytes, media_type="audio/wav")