import numpy as np
def crop_ar_frame(ar_frame,book_corners):
    tl, tr, br, bl = book_corners
    book_width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    book_height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    target_aspect = book_width / book_height
    src_h , src_w = ar_frame.shape[:2]
    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        new_w = int(src_h * target_aspect)
        x_start = (src_w - new_w) // 2
        cropped = ar_frame[:, x_start:x_start+new_w]
    else:
        new_h = int(src_w / target_aspect)
        y_start = (src_h - new_h) // 2
        cropped = ar_frame[y_start:y_start+new_h, :]
    return cropped