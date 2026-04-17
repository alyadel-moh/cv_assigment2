import cv2
import numpy as np
def get_book_corners_in_frame(H,book_img):
    h,w = book_img.shape[:2]
    corners = np.array([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ], dtype=np.float32)
    corners_h = np.hstack([corners, np.ones((4, 1))])
    projected_corners_h = (H @ corners_h.T).T
    projected_corners = projected_corners_h[:, :2] / projected_corners_h[:, 2:]
    return projected_corners
def draw_book_outline(frame, corners):
    """Draw the detected book quadrilateral on the frame."""
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    vis = frame.copy()
    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    for pt in corners.astype(int):
        cv2.circle(vis, tuple(pt), 5, (0, 0, 255), -1)
    return vis