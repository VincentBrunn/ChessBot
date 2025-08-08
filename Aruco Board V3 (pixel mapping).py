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

# Function to map (x, y) pixel to chess square
def pixel_to_square(x, y, squares):
    point = np.array([x, y])
    for square in squares:
        tl = square['tl']
        br = square['br']
        if all(tl <= point) and all(point <= br):
            return square['name']
    return None  # Outside board

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
            corners_array = corner[0]  # 4 corners of the marker

            # Correct corner of each marker based on physical placement
            corner_map = {
                1: corners_array[0],  # Marker 1 → A8 → top-left
                2: corners_array[3],  # Marker 2 → A1 → bottom-left
                3: corners_array[2],  # Marker 3 → H1 → bottom-right
                4: corners_array[1],  # Marker 4 → H8 → top-right
            }

            adjusted_point = corner_map[i]
            marker_positions[i] = adjusted_point

            # Visualize each corner point
            cv2.circle(frame, tuple(adjusted_point.astype(int)), 6, (0, 255, 255), -1)
            cv2.putText(frame, f"{marker_to_corner.get(i, '?')} (ID {i})",
                        tuple(adjusted_point.astype(int) + np.array([10, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # If all 4 markers are detected, draw the grid and label squares
        if all(k in marker_positions for k in [1, 2, 3, 4]):
            # Order: [A8, A1, H1, H8] → top-left, bottom-left, bottom-right, top-right
            top_left = marker_positions[1]
            bottom_left = marker_positions[2]
            bottom_right = marker_positions[3]
            top_right = marker_positions[4]

            # Precompute grid vectors
            grid_x = (top_right - top_left) / 8  # Across columns
            grid_y = (bottom_left - top_left) / 8  # Down rows

            squares = []  # Store all squares and positions

            # Draw vertical and horizontal lines
            for i in range(9):
                # Vertical lines
                start_v = top_left + i * grid_x
                end_v = bottom_left + i * grid_x
                cv2.line(frame, tuple(start_v.astype(int)), tuple(end_v.astype(int)), (0, 0, 255), 1)

                # Horizontal lines
                start_h = top_left + i * grid_y
                end_h = top_right + i * grid_y
                cv2.line(frame, tuple(start_h.astype(int)), tuple(end_h.astype(int)), (0, 0, 255), 1)


            # Label each square
            for row in range(8):
                for col in range(8):
                    square_tl = top_left + col * grid_x + row * grid_y
                    square_br = square_tl + grid_x + grid_y
                    square_center = square_tl + 0.5 * grid_x + 0.5 * grid_y

                    file = chr(ord('A') + col)
                    rank = str(8 - row)
                    square_name = file + rank

                    squares.append({
                        'name': square_name,
                        'center': square_center,
                        'tl': square_tl,
                        'br': square_br
                    })

                    # Draw square name at center
                    cv2.putText(frame, square_name,
                                tuple(square_center.astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Example usage: map mouse click to square
            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    sq = pixel_to_square(x, y, squares)
                    if sq:
                        print(f"Clicked on square: {sq}")

            cv2.setMouseCallback('Chessboard Grid Mapping', on_mouse)

    cv2.imshow('Chessboard Grid Mapping', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
