import cv2

# loads the trained algorithm for face detection
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#to capture video from webcam
webcam = cv2.VideoCapture(0)

while True:
    #read the current frame
    frame_read, frame = webcam.read()

    #convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_points = trained_face_data.detectMultiScale(grayscaled_img)
    print(face_points)
    
    #display with header
    cv2.imshow('Face Detector Harnoor', grayscaled_img)
    cv2.waitKey(1)
