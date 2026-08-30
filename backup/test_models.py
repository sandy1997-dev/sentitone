import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load LLM
model_name = "dheeyantra/dhee-nxtgen-qwen3-odia-v2"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load Sentiment Model (we'll use as fallback)
print("Loading sentiment model...")
sentiment_pipeline = pipeline("text-classification", model="Baps24/odia-sentiment-muril-v4")

# --- Rule-Based Emotion Detection ---
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
        # Fallback to model
        result = sentiment_pipeline(text)[0]
        if result['label'].lower() in ['negative', 'sad']:
            return "sad"
        elif result['label'].lower() in ['positive', 'happy']:
            return "happy"
        else:
            return "neutral"

# --- Test ---
test_input = "ମୋ ଆବେଦନ ଦୁଇ ମାସ ହେଲା ପେନ୍ଦା ହେଉନାହିଁ, ବହୁତ ଦୁଃଖିତ ଏବଂ ରାଗିତ"
emotion = get_emotion_rule_based(test_input)
print("Emotion detected:", emotion)

# --- Generate Response with Emotion-Aware Prompt ---
system_prompt = (
    "You are a warm, empathetic Odia language assistant for RTI (Right to Information) services in Odisha. "
    "You speak naturally, like a friendly call center agent. "
    "The user's current emotion is '{emotion}'. "
    "If the emotion is 'angry' or 'sad', apologize sincerely and reassure them that you will help. "
    "If the emotion is 'happy', respond warmly and positively. "
    "Keep responses short, conversational, and in Odia. "
    "Use phrases like 'ଆପଣ ଚିନ୍ତା କରନ୍ତୁ ନାହିଁ' (Don't worry) and 'ମୁଁ ଆପଣଙ୍କୁ ସାହାଯ୍ୟ କରିବି' (I will help you). "
    "Acknowledge their feelings before giving any information."
).format(emotion=emotion)

prompt = (
    f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
    f"<|im_start|>user\n{test_input}\n<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("\nLLM Response:")
print(response)