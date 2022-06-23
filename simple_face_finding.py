import face_recognition
import cv2
import os
import glob
import numpy as np

cam = cv2.VideoCapture(0)    # capturing video

fc_encoding = []
fc_name = []
frame_resizing = 0.25
def encoding_images(imgs_path):
    """Load encoding images from the given path.
    :path - imgs_path"""

    # Load images
    imgs_path = glob.glob(os.path.join(imgs_path, "*.*"))

    #print("{} encoding images found.".format(len(imgs_path)))

    # Store image encoding and names
    for img_path in imgs_path:
        img = cv2.imread(img_path)  # imread : loads images from the file
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get the filename from the file path.
        basename = os.path.basename(img_path)  # final component of the filename
        name =  os.path.splitext(basename)     # split the extenion of the filename
        
        # image encoding
        img_encoding = face_recognition.face_encodings(rgb_img)[0]

        # append the name and the encoded image(storing) 
        fc_encoding.append(img_encoding)
        fc_name.append(name)
        print(fc_name)

def detectKnowFaces(frame):
        small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(fc_encoding, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(fc_encoding, face_encoding)
            best_match_index = np.argmin(face_distances)  # Returns the indices of the minimum values along an axis.
            if matches[best_match_index]:
                name = fc_name[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        print(face_locations)
        face_locations = face_locations / frame_resizing
        return face_locations.astype(int), face_names








while True:
    ret, frame = cam.read() # Cam ON

    key = cv2.waitKey(1) 
    if key == ord('q'): # if 'q' is pressed it quit
        break
    cv2.imshow('frame', frame) # showing the frame 
    encoding_images('images/')
    detectKnowFaces(frame)
cam.release() # OFF cam
cv2.destroyAllWindows() 