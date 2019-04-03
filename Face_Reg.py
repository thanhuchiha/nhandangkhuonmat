#CHỤP ẢNH, LẤY DỮ LIỆU NGƯỜI DÙNG LÚC ĐẦU : Lấy 80 ảnh

#import requests
import cv2
import numpy as np
import database.ConnectionDB as db
import os

#url = "http://192.168.243.111:8080/shot.jpg"
cap = cv2.VideoCapture(0)  # mo cam
#cap.set (4, 100)
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # su dung ma nguon mo
id = 0
Name = input("Enter your name : ")
Class = input("Enter your class: ")
Gender = input("Gender : ")
Mssv = input("MSSV: ")
db.Insert_Student(Name,Class,Gender,Mssv)

sampleNum = 0
os.mkdir("dataset/"+Mssv)
while (True):
    ret, img = cap.read()  # doc cam

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # chuyen khong gian mau BGR COLOR_BGR2GRAY

    faces = faceDetect.detectMultiScale(gray, 1.3, 5);  # nhan dang khuon mat
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite("dataset/"+Mssv+"/" +str(sampleNum) + ".jpg", img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # ve hinh chu nhat xung quanh mat
        cv2.waitKey(100)
    cv2.imshow("face", img);
    cv2.waitKey(1)
    if (sampleNum > 50):
        break

cap.release()
cv2.destroyAllWindows()
