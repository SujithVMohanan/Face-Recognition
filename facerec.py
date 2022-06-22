import face_recognition
import cv2

img = cv2.imread("msd.jpeg")
#rgb_img = cv2.cvtColor(img2, cv2.COLOR_BAYER_BRG2RGB)
img_encoding = face_recognition.face_encodings(img)[0]



cv2.imshow("Img",img)
cv2.waitKey(0)