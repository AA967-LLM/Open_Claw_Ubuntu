
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import time
import asyncio

app = FastAPI()

model_path = "/root/Step3-VL-10B-model"
key_mapping = {
    "^vision_model": "model.vision_model",
    r"^model(?!\.(language_model|vision_model))": "model.language_model"
}

print("Loading model to CPU (fp16)...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cpu",
    torch_dtype=torch.float16,
    key_mapping=key_mapping
).eval()
print("Model loaded.")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    
    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cpu")
    
    # Generate
    if stream:
        async def event_generator():
            # Mock streaming for now or implement proper streamer
            # For simplicity, we'll do non-streaming logic and wrap it
            generate_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            response_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            chunk = {
                "id": "chatcmpl-" + str(int(time.time())),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "Step3-VL-10B",
                "choices": [{"index": 0, "delta": {"content": response_text}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        generate_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        return {
            "id": "chatcmpl-" + str(int(time.time())),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "Step3-VL-10B",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
