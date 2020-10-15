import cv2

#loads the trained algorithm for face detection
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect images
img = cv2.imread('image.jpg')

#convert image to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)




cv2.imshow('Face Detector Harnoor', grayscaled_img)
cv2.waitKey()

print("Code Completed")