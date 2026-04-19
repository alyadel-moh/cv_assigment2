import cv2
import numpy as np
import matplotlib.pyplot as plt
def compute_homography(pts1,pts2):
    n = len(pts1)
    A = []
    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A.append([-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
    A = np.array(A)
    _,_, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    H = H / H[2, 2] 
    return H
def verify_homography(H, pts1, pts2, img1, img2, num_pts=10):
    """
    Project a few pts1 through H and display them on img2.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Source points (img1)")
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Projected points on img2")

    colors = plt.cm.Set1(np.linspace(0, 1, num_pts))

    for i in range(num_pts):
        x, y = pts1[i]
        # Project
        p_h = H @ np.array([x, y, 1.0])
        px, py = p_h[0] / p_h[2], p_h[1] / p_h[2]

        axes[0].plot(x,  y,  'o', color=colors[i], markersize=8)
        axes[1].plot(px, py, 'x', color=colors[i], markersize=10, mew=2,
                     label=f"({px:.0f},{py:.0f})")
        axes[1].plot(pts2[i][0], pts2[i][1], 'o', color=colors[i],
                     markersize=8, alpha=0.5)

    axes[1].legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.show()