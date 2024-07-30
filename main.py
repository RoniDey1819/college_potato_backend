from fastapi import FastAPI, File, UploadFile
from uvicorn import run
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import requests
import gdown
import os

MODEL_FOLDER = 'temp_models'
MODEL_FILENAME = 'potato_blight_model_temp.h5'
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)  # Path to save the downloaded model

MODEL_ID = '19dFRIHQrnmrsH0SYdMiEsiMbxk_BYR-4'  # Google Drive file ID

# Create the folder if it doesn't exist
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Format the download URL
MODEL_URL = f'https://drive.google.com/uc?id={MODEL_ID}'

# Download the model from Google Drive if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Potato Early blight', 'Potato Late blight', 'Potato healthy']
print('Model loaded successfully!')

origins = [
    "*",
    "http://localhost:3000",
    "http://localhost",
    "https://potato-frontend-bgld.onrender.com/"
]




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Ping received successfully!"

async def read_file_as_image(data: bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image.tolist()  # Convert to list before returning

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = await read_file_as_image(image_bytes)
    
    img_batch=np.expand_dims(image, axis=0)
    
    predictions=MODEL.predict(img_batch)
    index=np.argmax(predictions[0])
    confidence=round(np.max(predictions[0])*100,4)
    print(CLASS_NAMES[index], confidence)
    
    return {"class":CLASS_NAMES[index],"confidence":confidence}





# if __name__ == "__main__":
#      run(app, host="localhost", port=80)
    
