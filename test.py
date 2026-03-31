import cv2
import numpy as np

def incremental_stitching(images: list) -> np.ndarray:
    if len(images) < 2:
        raise ValueError(f"Il faut au moins 2 images, reçu : {len(images)}")

    panorama = images[0].copy()

    for i, new_img in enumerate(images[1:], start=1):
        print(f"[STITCH] Ajout image {i}/{len(images)-1}...")

        # 1. Détection et matching de features
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(panorama, None)
        kp2, des2 = orb.detectAndCompute(new_img, None)

        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:50]

        # 2. Calcul de l'homographie
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

        # 3. Calcul de la taille exacte du canvas
        h1, w1 = panorama.shape[:2]
        h2, w2 = new_img.shape[:2]

        corners = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        corners_warped = cv2.perspectiveTransform(corners, H)
        all_corners = np.concatenate([
            np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2),
            corners_warped
        ])
        x_min, y_min = np.floor(all_corners[:,0,:].min(axis=0)).astype(int)
        x_max, y_max = np.ceil(all_corners[:,0,:].max(axis=0)).astype(int)

        # Translation pour rester en coordonnées positives
        T = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]], dtype=np.float64)

        # 4. Warp et collage
        canvas_w, canvas_h = x_max - x_min, y_max - y_min
        warped = cv2.warpPerspective(new_img, T @ H, (canvas_w, canvas_h))
        warped[-y_min:-y_min+h1, -x_min:-x_min+w1] = panorama
        panorama = warped

    return panorama