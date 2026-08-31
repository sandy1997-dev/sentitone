import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from indic_asr_onnx import IndicTranscriber
from typing import Optional
import io, tempfile, numpy as np, soundfile as sf, secrets, sqlite3, datetime, re
import torchaudio

# Patch torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    def _soundfile_load(path):
        data, sr = sf.read(path, dtype='float32', always_2d=True)
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
    c.execute('''CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE NOT NULL,
        client_name TEXT,
        created_at TEXT,
        is_active INTEGER DEFAULT 1
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS call_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_key TEXT,
        input_text TEXT,
        response_text TEXT,
        emotion TEXT,
        created_at TEXT
    )''')
    conn.commit()
    conn.close()
init_db()

# Load LLM (Gemma 2B - Odia fine-tuned, stable)
print("Loading LLM...")
model_name = "OdiaGenAI/odia_gemma_2b_base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load STT
print("Loading STT...")
stt_model = IndicTranscriber()

# TTS - Pre-recorded audio (works reliably)
RECORDINGS_DIR = "prompts"

VOICE_RECORDINGS = {
    "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିପାରିବି।": "greeting.wav",
    "ଆପଣଙ୍କ ଆବେଦନ ଏବେ ପ୍ରକ୍ରିୟାକରଣ ହେଉଛି।": "status.wav",
    "ମୁଁ ଦୁଃଖିତ, ଆପଣ ଚିନ୍ତା କରନ୍ତୁ ନାହିଁ।": "apology.wav",
    "ଆପଣଙ୍କୁ ଧନ୍ୟବାଦ!": "thanks.wav",
}

def generate_tts(text: str, voice: str = "female") -> bytes:
    if text in VOICE_RECORDINGS:
        path = os.path.join(RECORDINGS_DIR, VOICE_RECORDINGS[text])
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return f.read()
    for key, filename in VOICE_RECORDINGS.items():
        if key in text:
            path = os.path.join(RECORDINGS_DIR, filename)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return f.read()
    # Fallback silence
    audio = np.zeros(16000 * 2, dtype=np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
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

# Response cache
response_cache = {}

def generate_response(user_text):
    if user_text in response_cache:
        return response_cache[user_text]
    emotion = get_emotion(user_text)
    system_prompt = (
        f"You are a warm, empathetic Odia assistant for RTI services. The caller's emotion is '{emotion}'. "
        "If angry or sad, apologize and help. Keep responses short and conversational."
    )
    # Gemma chat format
    prompt = f"<start_of_turn>user\n{system_prompt}\n\nUser: {user_text}<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    response_cache[user_text] = response
    return response

def transcribe_audio(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        data, sr = sf.read(tmp_path, dtype='float32')
        if data.ndim > 1: data = data[:, 0]
        if sr != 16000:
            from scipy import signal
            new_len = int(len(data) * 16000 / sr)
            data_resampled = signal.resample(data, new_len)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp2:
                sf.write(tmp2.name, data_resampled, 16000, subtype='PCM_16')
                final_path = tmp2.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp2:
                sf.write(tmp2.name, data, 16000, subtype='PCM_16')
                final_path = tmp2.name
        text = stt_model.transcribe_ctc(final_path, "or")
    finally:
        os.remove(tmp_path)
        if 'final_path' in locals(): os.remove(final_path)
    return text

# FastAPI app
app = FastAPI(title="Sentitone API Gateway", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    message: str

class GenerateKeyRequest(BaseModel):
    admin_token: str = "admin_secret"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/admin/generate_key")
async def generate_key(request: GenerateKeyRequest):
    if request.admin_token != "admin_secret":
        raise HTTPException(status_code=403)
    new_key = secrets.token_hex(16)
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO api_keys (key, client_name, created_at, is_active) VALUES (?, 'default', ?, 1)", (new_key, datetime.datetime.now().isoformat()))
    conn.commit(); conn.close()
    return {"api_key": new_key}

@app.post("/chat")
async def chat(request: ChatRequest, api_key: str = Header(None)):
    if not api_key: raise HTTPException(status_code=401)
    response = generate_response(request.message)
    emotion = get_emotion(request.message)
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO call_logs (api_key, input_text, response_text, emotion, created_at) VALUES (?,?,?,?,?)", (api_key, request.message, response, emotion, datetime.datetime.now().isoformat()))
    conn.commit(); conn.close()
    return {"response": response, "emotion": emotion}

@app.post("/voice/stt")
async def stt(file: UploadFile = File(...), api_key: str = Header(None)):
    if not api_key: raise HTTPException(status_code=401)
    audio_bytes = await file.read()
    text = transcribe_audio(audio_bytes)
    return {"text": text}

@app.post("/voice/tts")
async def tts(text: str = Form(...), voice: str = Form("female"), api_key: str = Header(None)):
    if not api_key: raise HTTPException(status_code=401)
    audio_bytes = generate_tts(text, voice)
    return Response(content=audio_bytes, media_type="audio/wav")