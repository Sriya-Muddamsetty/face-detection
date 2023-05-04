import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    output = detector.detect_faces(frame)
    for face in output:
        x, y, w, h = face['box']

        left_eye_X , left_eye_Y = face['keypoints']['left_eye']
        right_eye_X , right_eye_Y = face['keypoints']['right_eye']
        nose_X , nose_Y = face['keypoints']['nose']
        mouth_left_X , mouth_left_Y = face['keypoints']['mouth_left']
        mouth_right_X , mouth_right_Y = face['keypoints']['mouth_right']

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (left_eye_X, left_eye_Y), 1, (0, 0, 255), 2)
        cv2.circle(frame, (right_eye_X, right_eye_Y), 1, (0, 0, 255), 2) #img, center, radius, color, thickness
        cv2.circle(frame, (nose_X, nose_Y), 1, (0, 0, 255), 2)
        cv2.circle(frame, (mouth_left_X, mouth_left_Y), 1, (0, 0, 255), 2)
        cv2.circle(frame, (mouth_right_X, mouth_right_Y), 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()