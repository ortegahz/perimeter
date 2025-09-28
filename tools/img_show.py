import os

import cv2
import numpy as np

img = np.full((200, 400, 3), 255, np.uint8)
cv2.putText(img, f"pid={os.getpid()}", (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("test-before", img)
cv2.waitKey(0)

from cores.byteTrackPipeline import ByteTrackPipeline

cv2.imshow("test-after", img)
cv2.waitKey(0)
