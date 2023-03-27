from imutils import face_utils
from imutils.face_utils import rect_to_bb
from flask import Flask, render_template
import argparse
import imutils
import dlib
import cv2
import mediapipe as mp
print("3333333333")
silhouette1= [10,21,54,58,67,93,103,109,127,132,136,148,149,150,152,162,172,176,234,251,284,
    288,297,323,332,338,356,361,365,377,378,379,389,397,400,454]
lipsUpperOuter = [0,37,39,40,61,185,267,269,270,291,409]
lipsLowerOuter = [17,84,91,146,181,291,314,321,375,405]
lipsUpperInner = [13,78,80,81,82,191,308,310,311,312,415]
lipsLowerInner = [14,78,87,88,95,178,308,317,318,324,402]
rightEyeUpper0 = [157,158,159,160,161,173,246]
rightEyeLower0 = [7,33,133,144,145,153,154,155,163]
rightEyeUpper1 = [27,28,29,30,56,190,247]
rightEyeLower1 = [22,23,24,25,26,110,112,130,243]
rightEyeUpper2 = [113,189,221,222,223,224,225]
rightEyeLower2 = [31,226,228,229,230,231,232,233,244]
rightEyeLower3 = [111,117,118,119,120,121,128,143,245]

rightEyebrowUpper = [55,63,66,70,105,107,156,193]
rightEyebrowLower = [35,46,52,53,65,124]

rightEyeIris = [473, 474, 475, 476, 477]

leftEyeUpper0 = [384,385,386,387,388,398,466]
leftEyeLower0 = [249,263,362,373,374,380,381,382,390]
leftEyeUpper1 = [257,258,259,260,286,414,467]
leftEyeLower1 = [252,253,254,255,256,339,341,359,463]
leftEyeUpper2 = [342,413,441,442,443,444,445]
leftEyeLower2 = [261,446,448,449,450,451,452,453,464]
leftEyeLower3 = [340,346,347,348,349,350,357,372,465]

leftEyebrowUpper = [285,293,296,300,334,336,383,417]
leftEyebrowLower = [265,276,282,283,295,353]

leftEyeIris = [468,469,470,471,472]

midwayBetweenEyes = [168]

noseTip = [1]
noseBottom = [2]
noseRightCorner = [98]
noseLeftCorner = [327]

rightCheek = [205]
leftCheek = [425]
# pip install cmake
# pip install dlib
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# help="path to input image")
# args = vars(ap.parse_args())
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner

from datetime import datetime
studet = "hhhhhh"
print("1111122222333333")
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", now=datetime.now())


if __name__=="__main__":
    app.run()
    
# @app.get("/get-by-name/{student_id}")
# def get_student(student_id: str, studet: str):
#     return {"student_id": "studet"}
#
# def live_video_processing():
#
#
#     lips = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270,
#             291, 291, 308, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
#
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(
#         "C:/Users/einav/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0/LocalCache/local-packages/Python310/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat")
#     fa = face_utils.FaceAligner(predictor, desiredFaceWidth=500)
#     # load the input image, resize it, and convert it to grayscale
#     # image = cv2.imread(args["image"])
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh()  # detect one person and give his position
#     mp_draw = mp.solutions.drawing_utils  # draw the points on the face
#     cap = cv2.VideoCapture(0)  # open the camera and take data in real time
#     drawSpec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)
#     test = 0
#     kkk = 0
#     while True:
#         test = test + 1
#         if test >10:
#             ret, img = cap.read()
#             results1 = face_mesh.process(img)
#             image = imutils.resize(img, width=800)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             # show the original input image and detect faces in the grayscale
#             # image
#             # cv2.imshow("Input", image)
#             rects = detector(gray, 2)
#             # loop over the face detections
#             for rect in rects:
#                 # extract the ROI of the *original* face, then align the face
#                 # using facial landmarks
#                 (x, y, w, h) = rect_to_bb(rect)
#                 faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
#                 faceAligned = fa.align(image, gray, rect)
#                 # display the output images
#                 # cv2.imshow("Original", faceOrig)
#                 # cv2.imshow("Aligned", faceAligned)
#                 results = face_mesh.process(faceAligned)
#         # draw points on the image
#                 if results.multi_face_landmarks:
#                     for face_landmarks in results.multi_face_landmarks:
#                         mp_draw.draw_landmarks(faceAligned, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
#                                                # mp_draw.DrawingSpec((0, 255, 0), 1, 1),
#                                                # mp_draw.DrawingSpec((0, 0, 255), 1, 1))  # in mp_draw.DrawingSpec() interface we can change the color
#                         for id, lm in enumerate(face_landmarks.landmark):
#                             if kkk< len(lips) and id == lips[kkk]:
#                                 kkk = kkk+1
#                                 print(id)  # print point id
#                                 print(lm)  # print x, y, z values
#                         kkk = 0
#             # print(results.multi_face_landmarks)
#                 cv2.imshow("Aligned", faceAligned)
#             cv2.imshow("Face mesh", img)  # open the camera
#             cv2.waitKey(1)
#             # if cv2.waitKey(1) & 0xFF == ord('q'):
#             #     break
#
#
#
# # image = cv2.imread('9.png')
# # image = imutils.resize(image, width=800)
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # # show the original input image and detect faces in the grayscale
# # # image
# # cv2.imshow("Input", image)
# # rects = detector(gray, 2)
# # # loop over the face detections
# # for rect in rects:
# # 	# extract the ROI of the *original* face, then align the face
# # 	# using facial landmarks
# # 	(x, y, w, h) = rect_to_bb(rect)
# # 	faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
# # 	faceAligned = fa.align(image, gray, rect)
# # 	# display the output images
# # 	cv2.imshow("Original", faceOrig)
# # 	cv2.imshow("Aligned", faceAligned)
# # 	mp_face_mesh = mp.solutions.face_mesh
# #
# # 	face_mesh = mp_face_mesh.FaceMesh()
# # 	height, width, _ = faceAligned.shape
# # 	rgb_image = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB)
# # 	result = face_mesh.process(faceAligned)
# # 	list_image_r = []
# # 	face_mesh = mp_face_mesh.FaceMesh()  # detect one person and give his position
# # 	mp_draw = mp.solutions.drawing_utils  # draw the points on the face
# # 	drawSpec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)
# #
# # 	for facial_landmarks in result.multi_face_landmarks:
# # 		for id, lm in enumerate(facial_landmarks.landmark):
# # 			x = lm.x
# # 			y = lm.y
# # 			z = lm.z
# # 			print(id)
# # 			shape = faceAligned.shape
# # 			print(x * width, ", ", y * height, ", ", z)
# # 			list_image_r.append((x * width, y * height, z))
# # 			print()
# # 			mp_draw.draw_landmarks(faceAligned, facial_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
# # 			cv2.imshow("Face Mesh1", faceAligned)
# # 	cv2.waitKey(0)
#
# if __name__ == "__main__":
#     live_video_processing()
#
