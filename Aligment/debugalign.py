# debug_align.py
import cv2
import numpy as np

img_path = "Dataset/image0000416.jpg"         # cambia qui
ref_path = "test_0015_aligned.jpg"          # cambia qui

img = cv2.imread(img_path)
ref = cv2.imread(ref_path)
ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=5000)  # aumenta per trovare più keypoints
kp1, des1 = orb.detectAndCompute(ref_gray, None)
kp2, des2 = orb.detectAndCompute(img_gray, None)
print("Keypoints ref:", len(kp1), "Keypoints img:", len(kp2))

if des1 is None or des2 is None:
    print("Descriptors missing in one image")
    raise SystemExit

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
knn = bf.knnMatch(des1, des2, k=2)
good = []
for m_n in knn:
    if len(m_n) != 2:
        continue
    m, n = m_n
    if m.distance < 0.85 * n.distance:  # puoi provare 0.75 -> 0.9
        good.append(m)
print("Good matches:", len(good))

if len(good) < 4:
    print("Too few matches for homography")
else:
    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts_img = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts_img, pts_ref, cv2.RANSAC, 5.0)
    print("H found:", H is not None)
    if H is not None:
        aligned = cv2.warpPerspective(img, H, (ref.shape[1], ref.shape[0]))
        cv2.imwrite("debug_aligned.jpg", aligned)
        matches_vis = cv2.drawMatches(ref, kp1, img, kp2, good[:50], None, flags=2)
        cv2.imwrite("debug_matches.jpg", matches_vis)
        print("Saved debug_aligned.jpg and debug_matches.jpg")