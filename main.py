from fastapi import FastAPI
import uvicorn
from starlette.responses import JSONResponse
from ai.funcs.image_preprocessing import image_preprocessing
from ai.funcs.model_prediction import get_prediction
import boto3
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# cors 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "crop doctor model api!"}


@app.post("/predict/")
async def predict(image_key: str):
    s3 = boto3.client('s3')

    # S3 객체 다운로드
    response = s3.get_object(Bucket='cropdoctor', Key=image_key)
    img_data = response['Body'].read()

    preprocessed_image = image_preprocessing(img_data)
    probability, solution_id = get_prediction(preprocessed_image)
    return JSONResponse({"solutionId": solution_id, "probability": probability})


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
