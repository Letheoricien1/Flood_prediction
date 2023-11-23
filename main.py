
import uvicorn ##ASG
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.applications import mobilenet_v3
import io

app = FastAPI()
classes =['Aphids: this pest is dangerious for corn and coffée ', 'Armyworms: this pest is dangerious for corn', 'Beetle: this pest is dangerious for coffee', 'Corn Bores: this pest is  dangerious for corn ','this pest is not dangerious for your crop']

class_model = tf.keras.models.load_model('MobileNet_92.h5')


def image_class(input_img):
    input_img = np.expand_dims(input_img, axis=0)
    input_img = mobilenet_v3.preprocess_input(input_img)
    predict_img = class_model.predict(input_img)
    if max(max(predict_img)) > 0.6:
        return classes[np.argmax(predict_img)]
    else:
        return classes[4]

@app.get("/")
async def root():
 return {"greeting":"Hello pest classification"}

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        img = img.resize((224, 224))  # Redimensionne l'image à la taille attendue par le modèle
        img_array = np.array(img)  # Convertit l'image en un tableau NumPy

        # Assure-toi que les dimensions sont correctes
        if img_array.shape != (224, 224, 3):
            return JSONResponse(content={"error": "Invalid image dimensions"})

        predicted_class = image_class(img_array)
        return JSONResponse(content={"class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)
#uvicorn main:app --reload"""

