import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn as nn
import torch.nn.modules.module as module

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form, Response
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.responses import HTMLResponse
from typing import Optional
import io, wave, tempfile, os, numpy as np, soundfile as sf, secrets, json, re
import torchaudio

# Patch torchaudio.load to use soundfile
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

# ---------- Load LLM ----------
model_name = "dheeyantra/dhee-nxtgen-qwen3-odia-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

# ---------- Load STT ----------
from indic_asr_onnx import IndicTranscriber
stt_model = IndicTranscriber()

# ---------- Emotion Detection (Rule-Based Only) ----------
def get_emotion_rule_based(text):
    sad_keywords = ["ଦୁଃଖ", "କାନ୍ଦ", "ନିରାଶ", "ବିଳମ୍ବ", "ଅସୁବିଧା", "ହେଉନାହିଁ", "ପେନ୍ଦା"]
    angry_keywords = ["ରାଗ", "ଗାଳି", "ଅଭିଯୋଗ", "ଚିଡ଼ା", "କାହିଁକି", "ହଟାତ"]
    happy_keywords = ["ଖୁସି", "ଧନ୍ୟବାଦ", "ଆନନ୍ଦ", "ସନ୍ତୋଷ"]
    
    if any(word in text for word in angry_keywords):
        return "angry"
    elif any(word in text for word in sad_keywords):
        return "sad"
    elif any(word in text for word in happy_keywords):
        return "happy"
    else:
        return "neutral"

# ---------- Response Cache ----------
response_cache = {
    "ମୋ ଆବେଦନ କେଉଁଠି ଅଛି?": "ଆପଣଙ୍କ ଆବେଦନ ଏବେ ପ୍ରକ୍ରିୟାକରଣ ହେଉଛି। ଦୟାକରି କିଛି ସମୟ ଅପେକ୍ଷା କରନ୍ତୁ।",
    "ମୋ ଆବେଦନ ବିଳମ୍ବ ହେଉଛି": "ମୁଁ ବୁଝିପାରୁଛି ଯେ ଆପଣ ନିରାଶ ହୋଇଛନ୍ତି। ଆପଣଙ୍କ ଆବେଦନ ବିଷୟରେ ମୁଁ ଯାଞ୍ଚ କରିଛି। ଦୟାକରି ଆଉ କିଛି ଦିନ ଅପେକ୍ଷା କରନ୍ତୁ।",
    "ଧନ୍ୟବାଦ": "ଆପଣଙ୍କୁ ସ୍ୱାଗତ! ଆଉ କିଛି ସହାୟତା ଦରକାର ହେଲେ ମୁଁ ଏଠାରେ ଅଛି।",
}

def generate_response(user_text):
    # Check cache first
    if user_text in response_cache:
        return response_cache[user_text]

    emotion = get_emotion_rule_based(user_text)
    system_prompt = (
        "You are a warm, empathetic Odia language assistant for RTI services in Odisha. "
        "You speak naturally, like a friendly call center agent. "
        f"The user's current emotion is '{emotion}'. "
        "If the emotion is 'angry' or 'sad', apologize sincerely and reassure them. "
        "If the emotion is 'happy', respond warmly. "
        "Keep responses short, conversational, and in Odia. "
        "Acknowledge their feelings before giving any information."
    )
    prompt = (
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_text}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.strip()
    # Cache the response
    response_cache[user_text] = response
    return response

# ---------- STT Helper ----------
def transcribe_audio(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        data, sr = sf.read(tmp_path, dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]
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
        if 'final_path' in locals():
            os.remove(final_path)
    return text

# ---------- API Key Management ----------
API_KEYS_FILE = "api_keys.json"

def load_api_keys():
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, 'r') as f:
            return set(json.load(f))
    else:
        return {"test_key_123"}

def save_api_keys(keys):
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(list(keys), f)

VALID_API_KEYS = load_api_keys()

# ---------- FastAPI Setup ----------
app = FastAPI(title="Sentitone – Odia Voice Agent API", version="2.0.0")

class ChatRequest(BaseModel):
    message: str

class GenerateKeyRequest(BaseModel):
    admin_token: str = "admin_secret"

# ---------- Dashboard ----------
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """<!DOCTYPE html>
<html>
<head><title>Sentitone API Tester</title></head>
<body style="font-family:Arial;margin:20px">
<h2>Sentitone Odia Voice API – Demo</h2>
<input id="key" placeholder="API Key" value="test_key_123">
<h3>Chat</h3>
<textarea id="msg" placeholder="Odia text here"></textarea><br>
<button onclick="chat()">Send Chat</button>
<h3>STT</h3>
<input type="file" id="audio" accept=".wav">
<button onclick="stt()">Transcribe</button>
<pre id="res"></pre>
<script>
async function chat(){
 const k=document.getElementById('key').value, m=document.getElementById('msg').value;
 const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json','api-key':k},body:JSON.stringify({message:m})});
 document.getElementById('res').innerText=JSON.stringify(await r.json(),null,2);
}
async function stt(){
 const k=document.getElementById('key').value, f=document.getElementById('audio').files[0];
 const d=new FormData(); d.append('file',f);
 const r=await fetch('/voice/stt',{method:'POST',headers:{'api-key':k},body:d});
 document.getElementById('res').innerText=JSON.stringify(await r.json(),null,2);
}
</script>
</body></html>"""

# ---------- Admin: Generate API Key ----------
@app.post("/admin/generate_key")
async def generate_api_key(request: GenerateKeyRequest):
    if request.admin_token != "admin_secret":
        raise HTTPException(status_code=403, detail="Invalid admin token")
    new_key = secrets.token_hex(16)
    VALID_API_KEYS.add(new_key)
    save_api_keys(VALID_API_KEYS)
    return {"api_key": new_key, "message": "Key generated successfully. Use this key in the 'api-key' header."}

# ---------- Chat Endpoint ----------
@app.post("/chat")
async def chat(request: ChatRequest, api_key: str = Header(None)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    response = generate_response(request.message)
    return {"response": response, "emotion": get_emotion_rule_based(request.message)}

# ---------- STT Endpoint ----------
@app.post("/voice/stt")
async def voice_stt(file: UploadFile = File(...), api_key: str = Header(None)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    audio_bytes = await file.read()
    text = transcribe_audio(audio_bytes)
    return {"text": text}

# ---------- TTS Endpoint (Disabled) ----------
@app.post("/voice/tts")
async def voice_tts(text: str = Form(...), voice: str = Form("female"), api_key: str = Header(None)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    raise HTTPException(status_code=501, detail="TTS is in development. Please use /chat and /voice/stt for now.")

# ---------- Voice Chat Endpoint (Disabled) ----------
@app.post("/voice/chat")
async def voice_chat(file: UploadFile = File(...), voice: str = Form("female"), api_key: str = Header(None)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    raise HTTPException(status_code=501, detail="Full voice chat coming with GPU deployment. Use /chat and /voice/stt separately.")