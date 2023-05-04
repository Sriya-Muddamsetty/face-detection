from mtcnn.mtcnn import MTCNN
import cv2

detector = MTCNN()
img = cv2.imread('images/h7.jpeg')

output = detector.detect_faces(img)
# for single face
# x, y, w, h = output[0]['box']
# left_eye_X , left_eye_Y = output[0]['keypoints']['left_eye']
# right_eye_X , right_eye_Y = output[0]['keypoints']['right_eye']
# nose_X , nose_Y = output[0]['keypoints']['nose']
# mouth_left_X , mouth_left_Y = output[0]['keypoints']['mouth_left']
# mouth_right_X , mouth_right_Y = output[0]['keypoints']['mouth_right']

# cv2.circle(img, (left_eye_X, left_eye_Y), 1, (0, 0, 255), 2)
# cv2.circle(img, (right_eye_X, right_eye_Y), 1, (0, 0, 255), 2) #img, center, radius, color, thickness
# cv2.circle(img, (nose_X, nose_Y), 1, (0, 0, 255), 2)
# cv2.circle(img, (mouth_left_X, mouth_left_Y), 1, (0, 0, 255), 2)
# cv2.circle(img, (mouth_right_X, mouth_right_Y), 1, (0, 0, 255), 2)


#for multiple faces
for i in range(len(output)):
    x, y, w, h = output[i]['box']
    left_eye_X , left_eye_Y = output[i]['keypoints']['left_eye']
    right_eye_X , right_eye_Y = output[i]['keypoints']['right_eye']
    nose_X , nose_Y = output[i]['keypoints']['nose']
    mouth_left_X , mouth_left_Y = output[i]['keypoints']['mouth_left']
    mouth_right_X , mouth_right_Y = output[i]['keypoints']['mouth_right']

    cv2.circle(img, (left_eye_X, left_eye_Y), 1, (0, 0, 255), 2)
    cv2.circle(img, (right_eye_X, right_eye_Y), 1, (0, 0, 255), 2) #img, center, radius, color, thickness
    cv2.circle(img, (nose_X, nose_Y), 1, (0, 0, 255), 2)
    cv2.circle(img, (mouth_left_X, mouth_left_Y), 1, (0, 0, 255), 2)
    cv2.circle(img, (mouth_right_X, mouth_right_Y), 1, (0, 0, 255), 2)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#cv2.rectangle(img, (output[0]['box'][0], output[0]['box'][1]), (output[0]['box'][0]+output[0]['box'][2], output[0]['box'][1]+output[0]['box'][3]), (255, 0, 0), 2)
cv2.imshow('window', img)
cv2.waitKey(0)

print(output)
#Output: [{'box': [125, 32, 23, 28], 'confidence': 0.999897837638855, 'keypoints': {'left_eye': (132, 42), 'right_eye': (143, 41), 'nose': (138, 48), 'mouth_left': (132, 52), 'mouth_right': (143, 51)}}]
