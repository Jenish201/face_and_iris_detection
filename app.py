# Importing libraries
import cv2
import matplotlib.pyplot as plt


# importing haaracascade for face detection
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# importing harracascade eye
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


# Set the video capture device
cap = cv2.VideoCapture(0)


# Create a while loop to detect the face in real time
while cap.isOpened():
    # read from the webcam
    ret, frame = cap.read()

    # convert the frame to grey-scale
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haarcascade parameter
    # detect the face
    face_detection = face_detector.detectMultiScale(img_grey, scaleFactor=1.3, minSize=(100,100), maxSize=(300,300))

    for (x,y,w,h) in face_detection:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,255),1)

    eye_detection = eye_detector.detectMultiScale(img_grey, scaleFactor=1.1, minSize=(30,30), maxSize=(50,50))

    for (x,y,w,h) in eye_detection:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255),1)
    
    # show the camera
    cv2.imshow('cam', frame)

    # stop the loop if the user enters q
    if cv2.waitKey(1) &  0xFF==ord('q'):
        break

# release the video capture device and destroy all windows that were created
cap.release()
cv2.destroyAllWindows()