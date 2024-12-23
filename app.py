from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ollama import generate
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to restrict origins as needed
    allow_credentials=True,
    allow_methods=["*"],  # You can restrict this to specific HTTP methods
    allow_headers=["*"],  # You can restrict this to specific headers
)

# Request model
class GenerateOptions(BaseModel):
    num_keep: Optional[int] = 5
    seed: Optional[int] = 42
    num_predict: Optional[int] = 100
    top_k: Optional[int] = 20
    top_p: Optional[float] = 0.9
    min_p: Optional[float] = 0.0
    typical_p: Optional[float] = 0.7
    repeat_last_n: Optional[int] = 33
    temperature: Optional[float] = 0.8
    repeat_penalty: Optional[float] = 1.2
    presence_penalty: Optional[float] = 1.5
    frequency_penalty: Optional[float] = 1.0
    mirostat: Optional[int] = 1
    mirostat_tau: Optional[float] = 0.8
    mirostat_eta: Optional[float] = 0.6
    penalize_newline: Optional[bool] = True
    stop: Optional[List[str]] = ["\n", "user:"]
    numa: Optional[bool] = False
    num_ctx: Optional[int] = 1024
    num_batch: Optional[int] = 2
    num_gpu: Optional[int] = 1
    main_gpu: Optional[int] = 0
    low_vram: Optional[bool] = False
    vocab_only: Optional[bool] = False
    use_mmap: Optional[bool] = True
    use_mlock: Optional[bool] = False
    num_thread: Optional[int] = 8

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    options: GenerateOptions

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        # Pass all the options to the ollama generate function
        response = generate(
            model=request.model,
            prompt=request.prompt,
            stream=request.stream,
            options=request.options.dict()  # Convert options to dict
        )

        # Debugging: Log the raw response
        print("Raw response:", response)

        # Return the entire response dictionary
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
