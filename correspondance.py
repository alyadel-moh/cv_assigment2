import cv2
import numpy as np
import matplotlib.pyplot as plt
def find_correspondences(image1, image2, n = 50):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n_match in raw_matches:
        if m.distance < 0.75 * n_match.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:n]
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    return points1, points2, keypoints1,keypoints2,good_matches

def plot_correspondences(img1, img2, kp1, kp2, good_matches):
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(16, 6))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title("50 Correspondences (SIFT + KNN + Ratio Test)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()