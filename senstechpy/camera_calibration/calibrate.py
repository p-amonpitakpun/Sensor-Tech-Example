import numpy as np
import cv2
import glob
import json

output_path = 'config.json'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

pattern_size = (7, 9)

images = glob.glob('checkboard/*.png')
n_img = len(images)
max_img = 200
images = np.random.choice(images, max_img, replace=False)

print("Checkboard Calibration")
print("found", n_img, "images")

if (n_img > 0):
    i = 1
    for fname in images:

        print("read image", fname, ">>", i, end="")

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            if len(corners) == (pattern_size[0] * pattern_size[1]):
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                print("\tcorners =", len(corners2))
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, pattern_size, corners2,ret)
                cv2.imshow('img',img)
                i += 1
                key = cv2.waitKey(15)
        else:
            print("\tnot found")

        if key == 24: break

    print("/*-------start calibration-------*/")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    h,  w = cv2.imread(images[0]).shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    print("\nmatrix")
    print(mtx)

    print("\ndist")
    print(dist)

    print("\nnew camera matrix")
    print(newcameramtx)

    print("\nROI")
    print(roi)

    tot_error = 0
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error

    err = mean_error/len(objpoints)
    print("\ntotal error: ", err)

    cv2.destroyAllWindows()

    config = dict()
    config['mtx'] = mtx.tolist()
    config['dist'] = dist.tolist()
    config['rvecs'] = [x.tolist() for x in rvecs]
    config['tvecs'] = [x.tolist() for x in tvecs]
    config['newcameramtx'] = newcameramtx.tolist()
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