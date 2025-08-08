import cv2
import cv2.aruco as aruco
import numpy as np

# ---------- Config ----------
BRIGHTNESS_THRESHOLD = 30
SAMPLE_SIZE = 5
calibrated = False
baseline_brightness = {}

# ---------- Helper Functions ----------
def measure_brightness(frame, center, sample_size=SAMPLE_SIZE):
    x, y = int(center[0]), int(center[1])
    half = sample_size // 2
    roi = frame[y - half:y + half + 1, x - half:x + half + 1]
    if roi.size == 0:
        return 0
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return int(np.mean(gray_roi))

def pixel_to_square(x, y, squares):
    point = np.array([x, y])
    for square in squares:
        tl = square['tl']
        br = square['br']
        if all(tl <= point) and all(point <= br):
            return square['name']
    return None

# ---------- Main Program ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

        # NEW: force exact corner index per marker
        fixed_corners = {
            1: 0,  # A1 (bottom-left)
            3: 3,  # H1 (bottom-right)
            2: 3,  # H8 (top-right)
            4: 1  # A8 (top-left)
        }

        for i, corner in zip(ids, corners):
            corners_array = corner[0]
            if i in fixed_corners:
                adjusted_point = corners_array[fixed_corners[i]]
                marker_positions[i] = adjusted_point

                # Draw visuals
                cv2.circle(frame, tuple(adjusted_point.astype(int)), 6, (0, 255, 255), -1)
                cv2.putText(frame, f"ID {i}", tuple(adjusted_point.astype(int) + np.array([10, -10])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Proceed only if all 4 markers are detected
        if all(k in marker_positions for k in [1, 2, 3, 4]):
            top_left = marker_positions[4]
            bottom_left = marker_positions[1]
            bottom_right = marker_positions[3]
            top_right = marker_positions[2]

            grid_x = (top_right - top_left) / 8
            grid_y = (bottom_left - top_left) / 8
            squares = []

            for row in range(8):
                for col in range(8):
                    tl = top_left + col * grid_x + row * grid_y
                    br = tl + grid_x + grid_y
                    center = tl + 0.5 * grid_x + 0.5 * grid_y
                    file = chr(ord('A') + col)
                    rank = str(8 - row)
                    name = file + rank
                    squares.append({'name': name, 'center': center, 'tl': tl, 'br': br})

            if not calibrated:
                cv2.putText(frame, "Press 'c' to calibrate baseline brightness (empty board).", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            else:
                occupied_squares = []

                for square in squares:
                    current = measure_brightness(frame, square['center'])
                    base = baseline_brightness.get(square['name'], 0)
                    delta = abs(current - base)

                    color = (0, 255, 0)
                    if delta > BRIGHTNESS_THRESHOLD:
                        color = (0, 0, 255)
                        occupied_squares.append(square['name'])
                        cv2.putText(frame, square['name'], tuple(square['center'].astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    cv2.circle(frame, tuple(square['center'].astype(int)), 4, color, -1)

                cv2.putText(frame, "Occupied: " + ", ".join(occupied_squares), (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Chessboard Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and ids is not None and all(k in marker_positions for k in [1, 2, 3, 4]):
        baseline_brightness = {
            square['name']: measure_brightness(frame, square['center'])
            for square in squares
        }
        calibrated = True
        print("Baseline brightness calibrated.")

cap.release()
cv2.destroyAllWindows()

