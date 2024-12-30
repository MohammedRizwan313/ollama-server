from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ollama import chat
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

class ChatRequest(BaseModel):
    query: str

class ChatResponseModel(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponseModel)
async def chat_with_model(request: ChatRequest):
    try:
        # Interact with the ollama chat model
        response = chat(model='llama3.1', messages=[
            {
                'role': 'user',
                'content': request.query,
            },
        ])

        # Return the response content
        return {"response": response['message']['content']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
