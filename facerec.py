import face_recognition
import cv2

def faceEncoding(filedir):
    img = cv2.imread(filedir)
    img_encoding = face_recognition.face_encodings(img)[0]
    # cv2.imshow("Img",img)
    # cv2.waitKey(0)
    return img_encoding

def facecomapre(encoded, encoded1):
    img_compare = face_recognition.compare_faces([encoded], encoded1)
    return img_compare


img1 = 'images/msd.jpeg'
img2 = 'images/msd1.jpg'
img3 = 'images/actor.jpg'
fcE1 = faceEncoding(img1)
fcE2 = faceEncoding(img2)
fcE3 = faceEncoding(img3)

print(facecomapre(fcE1, fcE2))
print(facecomapre(fcE2,fcE3))
print(facecomapre(fcE2, fcE1))