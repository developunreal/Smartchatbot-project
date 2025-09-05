# fastapi/main.py
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI(title="SmartChat (FastAPI → Ollama)")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "smollm2:135m")

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "SmartChat (FastAPI) is running."}

@app.post("/chat")
async def chat(req: ChatRequest):
    model = req.model or DEFAULT_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": req.message}],
        "stream": False
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Cannot reach Ollama at {OLLAMA_URL}: {e}")

    if resp.status_code != 200:
        # forward Ollama error for debugging
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    # Ollama non-streaming /api/chat returns `message` with `content`
    content = None
    if isinstance(data.get("message"), dict):
        content = data["message"].get("content")
    elif "response" in data:
        content = data.get("response")
    else:
        content = str(data)

    return {"reply": content, "ollama_raw": data}

