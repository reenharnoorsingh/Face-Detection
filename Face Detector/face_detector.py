import cv2

# loads the trained algorithm for face detection
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# choose an image to detect images
img = cv2.imread('image.jpg')

# convert image to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces
face_points = trained_face_data.detectMultiScale(grayscaled_img)
print(face_points)

# draw rectangles
for (x, y, w, h) in face_points:  # loops through all the faces in an image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


cv2.imshow('Face Detector Harnoor', img)
cv2.waitKey()

print("Code Completed")
