import cv2
import cv2.aruco as aruco
import numpy as np

# Marker ID to board corner mapping
marker_to_corner = {
    1: "A8",
    2: "A1",
    3: "H1",
    4: "H8"
}

# Initialize webcam
cap = cv2.VideoCapture(0)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        ids = ids.flatten()
        marker_positions = {}

        for i, corner in zip(ids, corners):
            center = corner[0].mean(axis=0)
            marker_positions[i] = center
            cv2.circle(frame, tuple(center.astype(int)), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"{marker_to_corner.get(i, '?')} (ID {i})",
                        tuple(center.astype(int) + np.array([10, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # If all 4 markers are detected, map grid
        if all(k in marker_positions for k in [1, 2, 3, 4]):
            pts = np.array([marker_positions[1], marker_positions[2],
                            marker_positions[3], marker_positions[4]], dtype=np.float32)

            # Order: [A8, A1, H1, H8] → top-left, bottom-left, bottom-right, top-right
            top_left, bottom_left, bottom_right, top_right = pts

            # Draw grid
            for i in range(9):
                # Vertical lines
                start = top_left + i * (top_right - top_left) / 8
                end = bottom_left + i * (bottom_right - bottom_left) / 8
                cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), (0, 0, 255), 1)

                # Horizontal lines
                start = top_left + i * (bottom_left - top_left) / 8
                end = top_right + i * (bottom_right - top_right) / 8
                cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), (0, 0, 255), 1)

    cv2.imshow('Chessboard Grid Mapping', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
