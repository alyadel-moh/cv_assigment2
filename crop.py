import numpy as np


def remove_black_borders(img, tol=10):
    # Create a mask of pixels that are NOT black
    mask = img > tol
    # If the image is color, check if any of the channels are not black
    if img.ndim == 3:
        mask = mask.any(axis=2)

    # Find the bounding box of the non-black content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Get the min/max coordinates
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return img[ymin : ymax + 1, xmin : xmax + 1]


def crop_ar_frame(ar_frame, book_corners):
    clean_frame = remove_black_borders(ar_frame)
    tl, tr, br, bl = book_corners
    book_width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    book_height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    target_aspect = book_width / book_height
    src_h, src_w = clean_frame.shape[:2]
    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        new_w = int(src_h * target_aspect)
        x_start = (src_w - new_w) // 2
        cropped = clean_frame[:, x_start : x_start + new_w]
    else:
        new_h = int(src_w / target_aspect)
        y_start = (src_h - new_h) // 2
        cropped = clean_frame[y_start : y_start + new_h, :]
    return cropped
