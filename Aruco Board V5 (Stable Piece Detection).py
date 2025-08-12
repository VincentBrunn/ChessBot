import cv2
import cv2.aruco as aruco
import numpy as np
from collections import deque, defaultdict

# =========================
# Tunables
# =========================
VIEW_W, VIEW_H = 800, 800      # canonical warped board size
INNER = 0.7                    # inner box fraction (avoid square borders)
SAMPLE_SIZE = 7                # still used in some ops (not critical now)

# Calibration & smoothing
CAL_FRAMES = 20
MEDIAN_WINDOW = 5              # temporal median per metric

# Brightness thresholds
MIN_ABS_BRIGHT = 12
BRIGHT_REL_FRAC = 0.12         # 12% relative difference
BRIGHT_MAD_K = 6.0             # robust threshold scale

# Edge/texture thresholds (Laplacian variance)
MIN_ABS_EDGE = 8.0
EDGE_MAD_K = 6.0

# Motion gate (pause decisions while hand is moving)
MOTION_GATE_THRESH = 8.0       # mean abs diff on warped view
MOTION_CLEAR_FRAMES = 6

DRAW_DEBUG = True

# Marker IDs & their touching corners (fixed physical corners)
# Your setup: 1=A1 (bottom-left), 3=H1 (bottom-right), 2=H8 (top-right), 4=A8 (top-left)
FIXED_CORNERS_INDEX = {
    1: 0,  # bottom-left of the marker
    3: 3,  # bottom-right
    2: 3,  # top-right
    4: 1,  # top-left
}

# =========================
# Helpers
# =========================
def inner_rect(rc, frac=INNER):
    (x0, y0, x1, y1) = rc
    w = x1 - x0
    h = y1 - y0
    mx = (x0 + x1) * 0.5
    my = (y0 + y1) * 0.5
    hw = (w * frac) * 0.5
    hh = (h * frac) * 0.5
    return int(mx - hw), int(my - hh), int(mx + hw), int(my + hh)

def square_rois(w, h):
    """Return per-square rects (tlx,tly,brx,bry) in warped space."""
    rois = {}
    cw = w / 8.0
    ch = h / 8.0
    for row in range(8):
        for col in range(8):
            x0 = int(col * cw)
            y0 = int(row * ch)
            x1 = int((col + 1) * cw)
            y1 = int((row + 1) * ch)
            name = f"{chr(ord('A') + col)}{8 - row}"
            rois[name] = (x0, y0, x1, y1)
    return rois

def measure_metrics(gray, rect):
    """Return (brightness_mean, edge_variance) in an inner region of rect."""
    x0, y0, x1, y1 = inner_rect(rect, INNER)
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(gray.shape[1], x1); y1 = min(gray.shape[0], y1)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0, 0.0
    b = float(np.mean(roi))
    # Laplacian variance as a texture/edge measure
    lap = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
    e = float(np.var(lap))
    return b, e

def robust_thresholds(samples):
    """Compute robust per-square thresholds for brightness and edge."""
    # samples[name] = list of (b,e)
    per_sq = {}
    for name, vals in samples.items():
        arr = np.array(vals, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            per_sq[name] = ((0.0, MIN_ABS_BRIGHT), (0.0, MIN_ABS_EDGE))
            continue
        b = arr[:, 0]; e = arr[:, 1]
        b_med = float(np.median(b))
        e_med = float(np.median(e))
        b_mad = float(np.median(np.abs(b - b_med))) + 1e-6
        e_mad = float(np.median(np.abs(e - e_med))) + 1e-6

        # Bright: combine absolute, relative, and MAD-based
        thr_b_abs = MIN_ABS_BRIGHT
        thr_b_rel = BRIGHT_REL_FRAC * max(1.0, b_med)
        thr_b_mad = BRIGHT_MAD_K * b_mad
        b_thr = float(max(thr_b_abs, thr_b_rel, thr_b_mad))

        # Edge: absolute floor + MAD-based
        thr_e_abs = MIN_ABS_EDGE
        thr_e_mad = EDGE_MAD_K * e_mad
        e_thr = float(max(thr_e_abs, thr_e_mad))

        per_sq[name] = ((b_med, b_thr), (e_med, e_thr))
    return per_sq

# =========================
# Main
# =========================
def main():
    # macOS-friendly backend
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    dst_quad = np.array([[0, 0], [VIEW_W - 1, 0], [VIEW_W - 1, VIEW_H - 1], [0, VIEW_H - 1]],
                        dtype=np.float32)

    rois = square_rois(VIEW_W, VIEW_H)

    # Calibration & smoothing
    calib_pairs = defaultdict(list)     # name -> list of (b,e)
    per_sq_base = {}                    # name -> ((b_base, b_thr), (e_base, e_thr))
    calibrated = False

    # Temporal median buffers
    recent_b = {n: deque(maxlen=MEDIAN_WINDOW) for n in rois}
    recent_e = {n: deque(maxlen=MEDIAN_WINDOW) for n in rois}

    # Motion gate
    last_warped_gray = None
    motion_quiet = 0
    motion_gate = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        pts = {}
        if ids is not None:
            ids = ids.flatten().tolist()
            for i, corner in zip(ids, corners):
                if i in FIXED_CORNERS_INDEX:
                    idx = FIXED_CORNERS_INDEX[i]
                    pt = corner[0][idx].astype(np.float32)
                    pts[i] = pt
                    if DRAW_DEBUG:
                        cv2.circle(frame, tuple(pt.astype(int)), 6, (0, 255, 255), -1)
                        cv2.putText(frame, f"ID {i}", tuple((pt + np.array([8, -8])).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        have_all = all(k in pts for k in [1, 2, 3, 4])

        if have_all:
            src = np.array([pts[4], pts[2], pts[3], pts[1]], dtype=np.float32)  # TL,TR,BR,BL
            M = cv2.getPerspectiveTransform(src, dst_quad)
            warped = cv2.warpPerspective(frame, M, (VIEW_W, VIEW_H))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            # Motion gate
            if last_warped_gray is None:
                last_warped_gray = gray.copy()
            diff = cv2.absdiff(gray, last_warped_gray).astype(np.float32)
            motion_score = float(np.mean(diff))
            last_warped_gray = gray

            if motion_score > MOTION_GATE_THRESH:
                motion_gate = True; motion_quiet = 0
            else:
                motion_quiet += 1
                if motion_quiet >= MOTION_CLEAR_FRAMES:
                    motion_gate = False

            # Calibration prompt
            if not calibrated:
                cv2.putText(frame, "Press 'c' to CALIBRATE (empty board).", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            occupied = []
            if calibrated and not motion_gate:
                vis = warped.copy()
                for name, rc in rois.items():
                    b, e = measure_metrics(gray, rc)
                    recent_b[name].append(b)
                    recent_e[name].append(e)
                    b_med = float(np.median(recent_b[name]))
                    e_med = float(np.median(recent_e[name]))

                    (b_base, b_thr), (e_base, e_thr) = per_sq_base[name]
                    bright_hit = abs(b_med - b_base) > b_thr
                    edge_hit = (e_med - e_base) > e_thr

                    occ = bright_hit or edge_hit
                    if occ:
                        occupied.append(name)

                    # Debug draw on warped view
                    if DRAW_DEBUG:
                        x0, y0, x1, y1 = inner_rect(rc, INNER)
                        color = (0, 0, 255) if occ else (0, 255, 0)
                        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
                        if occ:
                            cx = (x0 + x1) // 2; cy = (y0 + y1) // 2
                            cv2.putText(vis, name, (cx - 10, cy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                if DRAW_DEBUG:
                    small = cv2.resize(vis, (400, 400))
                    frame[0:400, 0:400] = small

                cv2.putText(frame, "Occupied: " + ", ".join(sorted(occupied)),
                            (10, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (240, 240, 240), 1)

            if motion_gate:
                cv2.putText(frame, "PAUSED (motion)", (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        else:
            if DRAW_DEBUG:
                cv2.putText(frame, "Need 4 ArUco corners visible.", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # UI / keys
        cv2.imshow("Chess Vision (hybrid robust)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('c') and have_all:
            # Gather CAL_FRAMES of (b,e) per square with empty board
            calib_pairs.clear()
            for _ in range(CAL_FRAMES):
                ok2, f2 = cap.read()
                if not ok2: break
                c2, i2, _ = aruco.detectMarkers(f2, aruco_dict, parameters=parameters)
                if i2 is None: continue
                i2 = i2.flatten().tolist()
                p2 = {}
                for j, crn in zip(i2, c2):
                    if j in FIXED_CORNERS_INDEX:
                        idx = FIXED_CORNERS_INDEX[j]
                        p2[j] = crn[0][idx].astype(np.float32)
                if all(k2 in p2 for k2 in [1, 2, 3, 4]):
                    src2 = np.array([p2[4], p2[2], p2[3], p2[1]], dtype=np.float32)
                    M2 = cv2.getPerspectiveTransform(src2, dst_quad)
                    w2 = cv2.warpPerspective(f2, M2, (VIEW_W, VIEW_H))
                    g2 = cv2.cvtColor(w2, cv2.COLOR_BGR2GRAY)
                    for nm, rc in rois.items():
                        b, e = measure_metrics(g2, rc)
                        calib_pairs[nm].append((b, e))
            per_sq_base.update(robust_thresholds(calib_pairs))
            # clear temporal buffers
            for nm in recent_b: recent_b[nm].clear()
            for nm in recent_e: recent_e[nm].clear()
            print("✅ Calibrated with brightness + edge baselines.")
            calibrated = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
