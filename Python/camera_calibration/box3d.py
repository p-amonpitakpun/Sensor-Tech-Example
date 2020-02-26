import numpy as np
import cv2
import json

config_path = "config.json"
pattern_size = (7, 9)

class Box3d:

    def __init__(self, pattern_size, config):
        self.pattern_size = pattern_size
        hp, wp = self.pattern_size

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        w, h, d = 6, 8, 6
        self.axis = np.float32([[0,0,0], [0,h,0], [w,h,0], [w,0,0], [0,0,-d],[0,h,-d],[w,h,-d],[w,0,-d] ])
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((hp * wp,3), np.float32)
        self.objp[:, :2] = np.mgrid[0:hp, 0:wp].T.reshape(-1, 2)

        self.mtx = np.array(config["newcameramtx"])
        self.dist = np.array(config["dist"])


    def draw(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size,None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
            
            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(self.objp, corners2, self.mtx, self.dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(self.axis, rvecs, tvecs, self.mtx, self.dist)

            
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, self.pattern_size, corners2, ret)
            
            
            imgpts = np.int32(imgpts).reshape(-1,2)
            # draw ground floor in green
            img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,255),-1)
            # draw pillars in blue color
            for i,j in zip(range(4),range(4,8)):
                img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),1)
            # draw top layer in red color
            img = cv2.drawContours(img, [imgpts[4:]],-1,(255,0,255),-1)

            return img

        else:
            return img

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    config = dict()

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    mtx = np.array(config["mtx"])
    newcameramtx = np.array(config["newcameramtx"])
    dist = np.array(config["dist"])

    box3d = Box3d(pattern_size, config)

    while(True):

        ret, frame = cap.read()

        if ret:
            
            # cv2.imshow("frame", frame)

            # frame = cv2.bilateralFilter(frame, 3, 25, 25)
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            x,y,w,h = roi
            img = cv2.undistort(frame, mtx, dist, None, newcameramtx)[y:y+h, x:x+w]

            img = box3d.draw(img)
            img = cv2.resize(img, (w * 2, h * 2))

            cv2.imshow("undistorted", img)
        
        key = cv2.waitKey(100 // 30)
        if key == 27: break

    cv2.destroyAllWindows()