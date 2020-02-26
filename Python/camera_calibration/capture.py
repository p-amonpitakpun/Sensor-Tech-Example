import cv2

cap = cv2.VideoCapture(0)

capturing = 0
index = 1
while(True):

    ret, frame = cap.read()

    if ret:
        cv2.imshow("videocapture", frame)
        if capturing:
            s_index = str(index)
            fname = "checkboard\\checkboard" + "0"*(4 - len(s_index)) + s_index + ".png"
            cv2.imwrite(fname, frame)
            print("written to", fname)
            index += 1

    key = cv2.waitKey(100 // 60)
    if key == 27 : 
        print(key)
        break
    elif key == 32:
        capturing = 1
        print("start capturing")