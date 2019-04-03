import numpy as np
import cv2
import requests

url = "http://192.168.1.34:8080/shot.jpg"
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)


    cv2.imshow('AndroidCam', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break