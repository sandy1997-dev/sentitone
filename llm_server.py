from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "dheeyantra/dhee-nxtgen-qwen3-odia-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")

@app.post("/v1/chat/completions")
async def chat(request: Request):
    data = await request.json()
    messages = data["messages"]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {"choices": [{"message": {"content": response}}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)