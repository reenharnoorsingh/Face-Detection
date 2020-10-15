import cv2

#loads the trained algorithm for face detection
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect images
img = cv2.imread('image.jpg')


print("Code Completed")