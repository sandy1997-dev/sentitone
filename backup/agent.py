import asyncio
import numpy as np
import wave
import torch
from livekit import rtc
from livekit.agents import Agent, AgentSession, ModelSettings, cli, stt, tts, WorkerOptions
from livekit.agents import utils
from livekit.plugins import silero
from indic_asr_onnx import IndicTranscriber
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Model Configuration ---
LLM_MODEL_NAME = "dheeyantra/dhee-nxtgen-qwen3-odia-v2"
SENTIMENT_MODEL_NAME = "Baps24/odia-sentiment-muril-v4"

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU (slow, not recommended for real-time)'}")

class OdiaRTIAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(
            instructions=(
                "You are an empathetic RTI assistant for Odisha. "
                "You speak fluent Odia. "
                "Always respond with compassion and clarity. "
                "If the user is frustrated, acknowledge their feelings before providing answers. "
            ),
            *args, **kwargs
        )

        print("Loading STT (IndicASR)...")
        self.stt_model = IndicTranscriber()

        print(f"Loading LLM: {LLM_MODEL_NAME}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print(f"Loading Sentiment Model: {SENTIMENT_MODEL_NAME}...")
        self.sentiment_pipeline = pipeline("text-classification", model=SENTIMENT_MODEL_NAME, device=device)

    def get_emotion_context(self, text):
        try:
            result = self.sentiment_pipeline(text)[0]
            sentiment = result['label']  # 'Positive', 'Negative', or 'Neutral'
            return sentiment.lower()
        except Exception:
            return "neutral"

    async def stt_node(self, audio_stream, model_settings):
        full_audio = await utils.audio.collect_audio(audio_stream)
        with wave.open("temp_audio.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(full_audio)
        text = self.stt_model.transcribe_ctc("temp_audio.wav", "or")
        yield stt.SpeechEvent(stt.SpeechEventType.FINAL_TRANSCRIPT, text)

    async def llm_node(self, chat_ctx, tools, model_settings):
        latest_user_msg = chat_ctx[-1].content if chat_ctx else ""

        emotion = self.get_emotion_context(latest_user_msg)

        emotion_prompt = (
            f"The user is currently feeling '{emotion}'. "
            "Adjust your tone accordingly: "
            "If feeling 'negative', show empathy and reassure them. "
            "If feeling 'positive', be warm and enthusiastic. "
        )

        prompt = f"<|im_start|>system\n{self.instructions}\n{emotion_prompt}<|im_end|>\n"
        for msg in chat_ctx:
            role = "user" if msg.role == "user" else "assistant"
            prompt += f"<|im_start|>{role}\n{msg.content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
        with torch.no_grad():
            outputs = self.llm_model.generate(**inputs, max_new_tokens=150)
        response = self.llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        yield response

    async def tts_node(self, text, model_settings):
        async for text_chunk in text:
            # Placeholder silence – will be replaced with IndicF5 later
            audio = np.zeros(16000 * 2, dtype=np.int16)
            frame = rtc.AudioFrame(
                data=audio.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(audio)
            )
            yield frame

async def entrypoint(ctx):
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=None,  # We override in Agent class, so not needed here
        llm=None,  # We override in Agent class
        tts=None,  # We override in Agent class
    )
    await session.start(room=ctx.room, agent=OdiaRTIAgent())
    # Removed session.say() – no TTS yet, so we skip the greeting.
    # The agent will respond when the user speaks.

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))