import face_recognition
import cv2

img = cv2.imread("msd.jpeg")
#rgb_img = cv2.cvtColor(img2, cv2.COLOR_BAYER_BRG2RGB)
img_encoding = face_recognition.face_encodings(img)[0]

img2 = cv2.imread("msd1.jpg")
#rgb_img = cv2.cvtColor(img2, cv2.COLOR_BAYER_BRG2RGB)
img_encoding2 = face_recognition.face_encodings(img2)[0]

img3 = cv2.imread("actor.jpg")
#rgb_img = cv2.cvtColor(img2, cv2.COLOR_BAYER_BRG2RGB)
img_encoding3 = face_recognition.face_encodings(img3)[0]


img_compare1 = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result : ", img_compare1)

img_compare2 = face_recognition.compare_faces([img_encoding], img_encoding3)
print("Result : ", img_compare2)

cv2.imshow("Img",img)
cv2.imshow("Img1", img2)
cv2.imshow("Img3", img3)
cv2.waitKey(0)