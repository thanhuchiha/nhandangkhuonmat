# USAGE
# python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# >=90% oke 8fps

# import the necessary packages
from imutils.video import VideoStream
#import PIL
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import database.ConnectionDB as db
#import requests
#from tkinter import *
#from PIL import Image, ImageTk

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=False,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding_model", required=False,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=False,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=False,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#Khởi tạo dữ liệu trực tiếp
args["detector"] = "face_detection_model"
args["embedding_model"] = "openface_nn4.small2.v1.t7"
args["recognizer"] = "output/recognizer.pickle"
args["le"] = "output/le.pickle"


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

check = False

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=900)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	#giaodien ben ngoai
	cv2.line(frame, (80, 0), (80, 700), (255, 255, 255), 1)  # height 700
	cv2.line(frame, (800, 0), (800, 700), (255, 255, 255), 1)
	cv2.line(frame, (0, 60), (900, 60), (255, 255, 255), 1)  # height 700
	cv2.line(frame, (0, 620), (900, 620), (255, 255, 255), 1)


	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# Tim kiem cac du lieu có trong database
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			file = le.classes_[j]
			name = file
			mssv = ""
			lop = ""
			if file != "unknown":
				args01 = db.Select_Student(file)
				name = args01[0]
				lop = args01[1]
				mssv = file



			text = "Name :{} {}".format(name,lop, proba * 100)

			y = startY - 10 if startY - 10 > 10 else startY + 10
			dochinhxac = "Accuracy :{:.2f}%".format(proba * 100)  # do chinh xac
			mssv01 = "MSSV: "+str(mssv)
			n = proba*100

			#Giao dien quanh khuon mat
			def RecTangle(Color):
				#(208, 148, 4)
				cv2.rectangle(frame, (startX, startY), (endX, endY), Color, 1) #chinh xac BGR
				cv2.line(frame, (startX, startY), (startX+80, startY), Color, 3)
				cv2.line(frame, (endX, startY), (endX-80, startY), Color, 3)
				#
				cv2.line(frame, (startX, endY), (startX + 80, endY), Color, 3)
				cv2.line(frame, (endX, endY), (endX - 80, endY), Color, 3)
				#
				cv2.line(frame, (startX, startY), (startX, startY+80), Color, 3)
				cv2.line(frame, (startX, endY), (startX, endY-80), Color, 3)
				#
				cv2.line(frame, (endX, startY), (endX, startY+80), Color, 3)
				cv2.line(frame, (endX, endY), (endX, endY-80), Color, 3)
				#
				cv2.line(frame, (startX-15, int((endY+startY)/2)), (startX+15, int((endY+startY)/2)), Color, 1)
				cv2.line(frame, (endX - 15, int((endY + startY) / 2)), (endX + 15, int((endY + startY) / 2)),Color, 1)
				#
				cv2.line(frame, (int((startX+endX)/2), startY-15), (int((startX+endX)/2), startY+15),Color, 1)
				cv2.line(frame, (int((startX + endX) / 2), endY - 15), (int((startX + endX) / 2), endY + 15),Color, 1)
			xacminh = "X"
			if file == "unknown": #Khong xac minh duoc
				RecTangle((0, 0, 255))
			elif n > 90: #Xac minh duoc
				RecTangle((0, 255, 0))
				xacminh = "V"

			#In ra các giá trị trên màn hình
			cv2.rectangle(frame, (startX,y-60),(endX,y+2),(208,148,4), -1)
			cv2.putText(frame, text, (startX, y-40),
				cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2)

			cv2.putText(frame, dochinhxac, (startX, y-20),
						cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2)
			if xacminh == "V":
				cv2.putText(frame, xacminh, (startX+170, y - 20),
						cv2.FONT_ITALIC, 0.5, (0, 255, 0), 4)
			else:
				cv2.putText(frame, xacminh, (startX + 170, y - 20),
							cv2.FONT_ITALIC, 0.5, (0, 0, 255), 4)
			cv2.putText(frame, mssv01, (startX, y),
						cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2)

			#

			#phat hien nguoi. luu neu > 90% thi tich ten vao csdl
			#print(text)




	# update the FPS counter
	fps.update()

	# show the output frame

	cv2.imshow("Face Recognition", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()