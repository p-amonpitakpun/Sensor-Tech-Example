import numpy as np
import cv2
import glob
import json

output_path = 'config.json'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('checkboard/*.png')
n_img = len(images)

print("Checkboard Calibration")
print("found", n_img, "images")

if (n_img > 0):
    i = 1
    for fname in images:

        print("read image", i)

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            cv2.imshow('img',img)
            key = cv2.waitKey(15)
            print("waitKey", key)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    print("new camera matrix")
    print(newcameramtx)

    print("ROI")
    print(roi)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error

    err = mean_error/len(objpoints)
    print("total error: ", err)

    cv2.destroyAllWindows()

    config = dict()
    config['mtx'] = mtx
    config['dist'] = dist
    config['rvecs'] = rvecs
    config['tvecs'] = tvecs
    config['newcameramtx'] = newcameramtx
    config['roi'] = roi
    config['err'] = err

    with open(output_path, 'w') as output_file:
        print('write output file', output_path)
        try:
            json.dump(config, output_file)
        except Exception as e:
            print('error:', e)

    print("finish calibrating")

else:
    exit(1)