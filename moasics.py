import cv2
import numpy as np
import matplotlib.pyplot as plt

def warp_image(img, H):
    h, w = img.shape[:2]
    
    # Step 1: map the 4 corners of img through H to find output bounds
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners_h = np.column_stack([corners, np.ones(4)])  # homogeneous
    mapped = (H @ corners_h.T).T
    mapped /= mapped[:, 2:3]  # divide by w to get 2D coords
    mapped_2d = mapped[:, :2]

    # Step 2: compute bounding box in destination coordinate system
    x_min, y_min = np.floor(mapped_2d.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(mapped_2d.max(axis=0)).astype(int)

    out_w = x_max - x_min
    out_h = y_max - y_min

    # Step 3: inverse warp — for each output pixel, find source in img
    H_inv = np.linalg.inv(H)
    warped = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # build grid of output pixel coordinates
    ys, xs = np.mgrid[0:out_h, 0:out_w]
    
    # Shift xs and ys to their absolute coordinates in the destination space.
    xs_absolute = xs + x_min
    ys_absolute = ys + y_min

    # apply H_inv to get source coords in img1
    ones = np.ones_like(xs_absolute)
    coords = np.stack([xs_absolute, ys_absolute, ones], axis=-1).reshape(-1, 3).T
    src_coords = (H_inv @ coords.astype(np.float64))
    src_coords /= src_coords[2:3, :]
    src_x = src_coords[0].reshape(out_h, out_w)
    src_y = src_coords[1].reshape(out_h, out_w)

    # Step 4: bilinear interpolation
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # fractional distances
    dx = src_x - x0
    dy = src_y - y0

    # FIXED: Changed (x1 < w) to (x0 < w) to prevent dropping the right/bottom-most edge.
    valid = (x0 >= 0) & (y0 >= 0) & (x0 < w) & (y0 < h)

    for c in range(3):
        top_left     = np.where(valid, img[np.clip(y0,0,h-1), np.clip(x0,0,w-1), c], 0)
        top_right    = np.where(valid, img[np.clip(y0,0,h-1), np.clip(x1,0,w-1), c], 0)
        bot_left     = np.where(valid, img[np.clip(y1,0,h-1), np.clip(x0,0,w-1), c], 0)
        bot_right    = np.where(valid, img[np.clip(y1,0,h-1), np.clip(x1,0,w-1), c], 0)

        warped[:, :, c] = np.where(
            valid,
            (top_left  * (1-dx) * (1-dy) +
             top_right * dx  * (1-dy) +
             bot_left  * (1-dx) * dy  +
             bot_right * dx  * dy),
            0
        ).astype(np.uint8)

    # FIXED: Return the absolute coordinates instead of conditional offsets
    return warped, x_min, y_min


def create_mosaic(img1, img2, H):
    warped_img1, x_min, y_min = warp_image(img1, H)

    h1, w1 = warped_img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate the global canvas boundaries
    canvas_x_min = min(0, x_min)
    canvas_y_min = min(0, y_min)
    
    canvas_x_max = max(w2, x_min + w1)
    canvas_y_max = max(h2, y_min + h1)

    canvas_w = canvas_x_max - canvas_x_min
    canvas_h = canvas_y_max - canvas_y_min

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Calculate the relative placement offsets
    offset_x1 = x_min - canvas_x_min
    offset_y1 = y_min - canvas_y_min
    
    offset_x2 = 0 - canvas_x_min
    offset_y2 = 0 - canvas_y_min
    
    canvas[offset_y1:offset_y1+h1, offset_x1:offset_x1+w1] = warped_img1
    canvas[offset_y2:offset_y2+h2, offset_x2:offset_x2+w2] = img2

    return canvas