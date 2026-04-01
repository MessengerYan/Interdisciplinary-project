"""
➵ Authors : Mael Pierron, Jean-Max Agogué
➵ Date : 17/03/2026
➵ Objective : Upgrade of the stitched image quality
"""
import cv2
import numpy as np

def postprocessing(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    heights = [0] * w
    best_area = 0
    best_rect = (0, 0, 0, 0)
    for i in range(h):
        for j in range(w):
            if mask[i][j] == 1:
                heights[j] += 1
            else:
                heights[j] = 0
        stack = []
        j = 0
        while j <= w:
            curr_height = heights[j] if j < w else 0
            if not stack or curr_height >= heights[stack[-1]]:
                stack.append(j)
                j += 1
            else:
                top = stack.pop()
                width = j if not stack else j - stack[-1] - 1
                area = heights[top] * width
                if area > best_area:
                    best_area = area
                    x = stack[-1] + 1 if stack else 0
                    y = i - heights[top] + 1
                    best_rect = (x, y, width, heights[top])
    x, y, w, h = best_rect
    return image[y:y+h, x:x+w]