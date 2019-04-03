#TRÍCH XUẤT NHÚNG. PHÂN TÍCH. CHUYỂN DỮ LIỆU VỀ DẠNG NHẬN DẠNG KHUÔN MẶT
#trích xuất tính năng học sâu để tạo ra một vectơ 128-D mô tả khuôn mặt

# USAGE
# python extract_embeddings.py --dataset dataset --embeddings output/embeddings.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7\
#	--detector face_detection_model --embedding-model openface_nn4.small2.v1.t7

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parser and parse the arguments phân tích cú pháp và phân tích đối số
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=False,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--embeddings", required=False,
	help="path to output serialized db of facial embeddings")
ap.add_argument("-d", "--detector", required=False,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding_model", required=False,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#khởi tạo dữ liệu trực tiếp
args["detector"] = "face_detection_model"
args["dataset"] = "dataset"
args["embeddings"] = "output/embeddings.pickle"
args["embedding_model"] = "openface_nn4.small2.v1.t7"

# load our serialized face detector from disk Tải cái nhận dạng khuôn mặt trực tiếp so sánh
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk Tải mô hình nhúng khuôn mặt nối tiếp
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# grab the paths to the input images in our dataset -  Lấy hình ảnh đến đường dẫn đầu vào trong tập dữ liệu đã có
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize our lists of extracted facial embeddings and - Khởi tạo danh sách và trích xuât khuôn mặt
# corresponding people names - Tên người tương ứng
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0

# loop over the image paths - Khởi tạo số lương khuôn mặt được xử lý
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path - Trích xuất tên người từ đường dẫn hình ảnh
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the image, resize it to have a width of 600 pixels (while - Thay đổi kích thước hình ảnh để có kích thước 600px
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image - Xây dựng 1 imageBlod từ thư mục hình ảnh
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize - Ap dụng opencv deep learning để dò tìm mã hóa
	# faces in the input image - Khuôn mặt trong hình ảnh đầu vào
	detector.setInput(imageBlob)
	detections = detector.forward()

	# ensure at least one face was found - Đảm bảo ít nhất 1 khuôn mặt được tìm thấy
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE - Đưa ra giả định là mỗi hình ảnh chỉ có 1 người
		# face, so find the bounding box with the largest probability - Khuôn mặt, tìm giới hạn xác suất lớn nhất
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also - Đảm bảo việc phát hiện với xác suất lớn nhất
		# means our minimum probability test (thus helping filter out - Thử nghiệm xác suất tối thiểu
		# weak detections) - Phát hiện yếu
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for - Tính toán tọa độ (x,y) cho khung giới hạn
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions - Trích xuất ROI và lấy kích thước ROI
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large - Đảm bảo chiều rộng và cao của khuôn mặt đủ lớn
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob - Xây dựng 1 BLOD cho ROI, SAU ĐÓ VƯỢT QUA BLOD
			# through our face embedding model to obtain the 128-d - Thông qua mô hình nhúng khuôn mặt để có được 128d
			# quantification of the face - định lượng khuôn mặt
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# add the name of the person + corresponding face - Thêm tên người và khuôn mặt tương ứng
			# embedding to their respective lists - Nhúng vào danh sách tương ứng
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# dump the facial embeddings + names to disk - Đổ các nhúng vào đĩa
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()