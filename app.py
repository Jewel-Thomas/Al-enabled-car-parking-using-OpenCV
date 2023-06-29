from flask import Flask, Response
from flask_cors import CORS
import cv2
import pickle
import cvzone
import numpy as np

width, height = 107,48

cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

def CheckParkingSpace(imgPro, img):
    spaceCounter = 0
    for pos in posList:
        x,y = pos
        imgCrop = imgPro[y:y+height,x:x+width]
        count = cv2.countNonZero(imgCrop)

        if count<900:
            color = (0,255,0)
            thickness = 4
            spaceCounter+=1
        else:
            color = (0, 0, 255)
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 60), scale=4,
                       thickness=5, offset=20, colorR=(0, 200, 0))

def generate():
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)

        success, img = cap.read()
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV,25,16)

        imgMedian = cv2.medianBlur(imgThreshold,5)
        kernel = np.ones((3,3),np.uint8)
        imgDilate = cv2.dilate(imgMedian,kernel,iterations=1)

        CheckParkingSpace(imgDilate, img)

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

app = Flask(__name__)
CORS(app)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)