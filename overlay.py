import cv2
import numpy as np
from homography import compute_homography
import matplotlib.pyplot as plt


def overlay_ar_frame(main_frame, cropped_ar_frame, target_corners):
    ar_h, ar_w = cropped_ar_frame.shape[:2]
    main_h, main_w = main_frame.shape[:2]

    # Define 4 corners of cropped AR frame
    src_corners = np.array(
        [[0, 0], [ar_w - 1, 0], [ar_w - 1, ar_h - 1], [0, ar_h - 1]], dtype=np.float32
    )

    dst_corners = target_corners.astype(np.float32)

    # Compute  Homography matrix & warps it
    H_overlay = compute_homography(src_corners, dst_corners)
    warped_ar = cv2.warpPerspective(cropped_ar_frame, H_overlay, (main_w, main_h))

    # Create mask to hollow the book area
    mask = np.zeros((main_h, main_w), dtype=np.uint8)
    cv2.fillPoly(mask, [dst_corners.astype(np.int32)], 255)

    mask_inv = cv2.bitwise_not(mask)
    main_bg = cv2.bitwise_and(main_frame, main_frame, mask=mask_inv)

    return cv2.add(main_bg, warped_ar)


def display_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(rgb_frame)
    plt.title("Frame Overlayed")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
