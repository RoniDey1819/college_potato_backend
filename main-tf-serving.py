from fastapi import FastAPI, File, UploadFile, HTTPException
from uvicorn import run
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "http://localhost",
]

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
endpoint = "http://localhost:8501/v1/models/potato_models:predict"  # Update with your endpoint

CLASS_NAMES = ['Potato Early blight', 'Potato Late blight', 'Potato healthy']

@app.get("/ping")
async def ping():
    return "Ping received successfully!"

async def read_file_as_image(data: bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image.tolist()  # Convert to list before returning

@app.post("/predict")
async def predict(img_file: UploadFile = File(...)):
    image_bytes = await img_file.read()
    image = await read_file_as_image(image_bytes)
    
    img_batch = np.expand_dims(image, axis=0)
    
    json_data = {
        "instances": img_batch.tolist()
    }
    
    try:
        response = requests.post(endpoint, json=json_data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        predictions = response.json()["predictions"][0]
        index = np.argmax(predictions)
        confidence = round(predictions[index] * 100, 4)
        class_name = CLASS_NAMES[index]
        return {"class": class_name, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    run(app, host="localhost", port=8080)
