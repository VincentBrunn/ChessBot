import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)

# Use getPredefinedDictionary instead of Dictionary_get
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow('Aruco Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
