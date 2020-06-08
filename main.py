from fastapi import FastAPI, File, UploadFile
import uvicorn
from starlette.requests import Request
import numpy as np

import io
import json
import os

import const                        # const.py
from detect import detect_face      # detect/detect_face.py


const.FOLDER_CURRENT = os.path.dirname(os.path.abspath(__file__))
const.FOLDER_CASCADES = os.path.join(const.FOLDER_CURRENT,'cascades')
const.PATH_FACE_DETECTOR = os.path.join(const.FOLDER_CURRENT, const.FOLDER_CASCADES + '/haarcascade_frontalface_default.xml')
const.SAMPLE_IMG_PATH = os.path.join(const.FOLDER_CURRENT, 'women.jpg')


app = FastAPI()

@app.post("/predict/face")
async def predict(request: Request, file: bytes = File(...)):
    data = {"success": False}

    if request.method == "POST":
        data = detect_face.detect_face(file)

    return data


# You can connect the debugger in your editor, for example with Visual Studio Code or PyCharm.
# python main.py
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
