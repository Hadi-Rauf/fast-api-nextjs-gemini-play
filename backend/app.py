# main.py
import uuid
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import google.generativeai as palm
import random
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from gloabl_ver import questions  # Ensure this is correctly spelled

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("BARD_API_KEY")
if not api_key:
    raise ValueError("gemini key not found. Make sure it's set in the .env file.")

folder_path = "FastAPI_sessions"

defaults = {
    'model': 'models/text-bison-001',
    'temperature': 0.1,
    'candidate_count': 1,
    'top_k': 40,
    'top_p': 0.95,
    'max_output_tokens': 1024,
    'stop_sequences': [],
    'safety_settings': [{"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_TOXICITY", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_VIOLENCE", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_MEDICAL", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"}],
}

palm.configure(api_key=api_key)

file_path = 'stored_data.json'

class QARequest(BaseModel):
    question: str
    answer: str

class PredictionRequest(BaseModel):
    data: dict

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI Mental Health API"}

@app.get("/generate_session")
async def generate_session():
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    session_id = str(uuid.uuid4())
    filename = f"{folder_path}/{session_id}.json"
    data = []
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return {"session_id": session_id}

@app.get("/generate_questions/{session_id}")
async def generate_questions(session_id: str):
    if not os.path.exists(f"{folder_path}/{session_id}.json"):
        raise HTTPException(status_code=404, detail="Invalid session ID")
    number_of_questions = 10
    ask_question = random.sample(questions, number_of_questions)
    return {"questions": ask_question}

@app.post("/store_qa/{session_id}")
async def store_qa(session_id: str, qa_request: QARequest):
    if not os.path.exists(f"{folder_path}/{session_id}.json"):
        raise HTTPException(status_code=404, detail="Invalid session ID")

    with open(f"{folder_path}/{session_id}.json", 'r') as file:
        existing_data = json.load(file)
    
    existing_data.append(qa_request.dict())

    with open(f"{folder_path}/{session_id}.json", 'w') as file:
        json.dump(existing_data, file, indent=4)

    return {"status": "success", "data": existing_data}

@app.post("/predict")
async def predict(prediction_request: PredictionRequest):
    try:
        data = prediction_request.data
        prompt = f"""You are an expert coder and programmer. You can analyze coding skills. Using the given data: {data}, the response should include:
        {{
            'finally': 'expert' or 'good' or 'need practice',
            'reason': 'Predict the detailed reason for the result.',
            'things to do': 'Predict detailed actions to be taken based on the result.'
        }}."""

        completion = palm.generate_text(
            **defaults,
            prompt=prompt,
        )
        response = completion.result

        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_string = response[json_start:json_end]

        json_string = json_string.replace('\\"', '"').replace("\n", "").replace("\\", "")
        final_response = json.loads(json_string)
        return final_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
