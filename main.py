from fastapi import FastAPI
import uvicorn
from starlette.responses import JSONResponse
from funcs.image_preprocessing import image_preprocessing
from funcs.model_prediction import get_prediction
import boto3
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
load_dotenv()


app = FastAPI()

# cors 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3001", "http://kdt-ai6-team03.elicecoding.com:3001"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "crop doctor model api!"}


@app.post("/predict")
async def predict(image_key: str):
    endpoint_url = os.getenv('endpoint_url')
    access_key = os.getenv('aws_access_key_id')
    secret_key = os.getenv('aws_secret_access_key')
    s3 = boto3.client('s3', endpoint_url=endpoint_url, aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key)

    # S3 객체 다운로드
    response = s3.get_object(Bucket='cropdoctor', Key=image_key)
    img_data = response['Body'].read()

    preprocessed_image = image_preprocessing(img_data)
    probability, solution_id = get_prediction(preprocessed_image)
    return JSONResponse({"solutionId": solution_id, "probability": probability})


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
