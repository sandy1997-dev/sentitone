# Sentitone – Odia Voice AI API

**Sentitone** is a Sarvam-like AI platform specializing in Odia language services. It provides:

- **Text Chat (LLM)** – Understands Odia text and generates empathetic responses.
- **Speech-to-Text (STT)** – Converts Odia audio to text.
- **(Coming Soon) Text-to-Speech (TTS)** – Converts Odia text to speech.

## 🔑 Getting an API Key

1. Call the admin endpoint to generate a key:

```bash
curl -X POST http://your-server:8000/admin/generate_key \
  -H "Content-Type: application/json" \
  -d '{"admin_token": "admin_secret"}'