import numpy as np
import cv2
import json


cap = cv2.VideoCapture(0)

config = dict()

with open("config.json", "r") as config_file:
    config = json.load(config_file)

mtx = np.array(config["mtx"])
newcameramtx = np.array(config["newcameramtx"])
dist = np.array(config["dist"])

while(True):

    ret, frame = cap.read()

    if ret:
        
        cv2.imshow("frame", frame)
                
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        x,y,w,h = roi
        img = cv2.undistort(frame, mtx, dist, None, newcameramtx)[y:y+h, x:x+w]

        cv2.imshow("undistorted", img)
    
    key = cv2.waitKey(100 // 60)
    if key == 27: break

cv2.destroyAllWindows()