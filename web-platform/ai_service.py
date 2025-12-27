"""
Minimal AI inference service.
No database, no auth - just AI model inference.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import random

app = FastAPI(
    title="Edge AI Inference Service",
    description="Stateless AI inference endpoint",
    version="1.0.0"
)

class InferenceRequest(BaseModel):
    """AI inference request."""
    message: str
    images: Optional[List[str]] = None  # Base64 data URLs
    temperature: float = 0.7
    max_tokens: int = 150

class InferenceResponse(BaseModel):
    """AI inference response."""
    content: str
    debug: Optional[dict] = None

@app.post("/api/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Run AI inference on the provided input.
    
    In production, this would:
    - Load your LLM/VLM model
    - Process text and images
    - Return model output
    
    For now, returns mock responses.
    """
    # Simulate processing time
    await asyncio.sleep(0.5 + random.random() * 1.5)
    
    # Mock AI response (replace with actual model inference)
    responses = [
        "I'm processing your request on the edge device. The model is analyzing your input...",
        "Based on the data processed locally, here's what I found...",
        "Running inference on the edge hardware. This keeps your data private and secure.",
        "The edge AI model has completed processing. Here are the results...",
        "Processing complete. The advantage of edge computing is the low latency you're experiencing."
    ]
    
    content = random.choice(responses)
    
    if request.images:
        content += f" I can see you've shared {len(request.images)} image(s) with me."
    
    debug_info = {
        "model": "mock-llm-v1",
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "prompt_tokens": len(request.message),
        "completion_tokens": len(content),
    }
    
    return InferenceResponse(
        content=content,
        debug=debug_info
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-inference"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
