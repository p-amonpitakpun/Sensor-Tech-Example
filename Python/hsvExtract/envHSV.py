import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def getImage():
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    ret, img = cap.read()
    time.sleep(1)    

    ret, img = cap.read()
    return  img

if __name__ == "__main__":

    env = getImage()
    cv2.imshow("environment", env)

    h, w, c = env.shape

    print("| environment")
    print("| >> \t shape =", env.shape)

    env_hsv = cv2.cvtColor(env, cv2.COLOR_BGR2HSV)

    envs = cv2.split(env_hsv)

    fig, axs = plt.subplots(c, 1, figsize=(5, 5))

    fig, ax = plt.subplots(figsize=(5, 5))

    x = envs[0].reshape((-1, ))
    axs[0].hist(x, color='blue', bins=100, alpha=0.3, label='H')

    x = envs[1].reshape((-1, ))
    axs[1].hist(x, color='blue', bins=100, alpha=0.3, label='S')

    x = envs[2].reshape((-1, ))
    axs[2].hist(x, color='blue', bins=100, alpha=0.3, label='V')

    axs[0].set_ylim(0, max(x))
    axs[0].legend()
    
    axs[1].set_ylim(0, max(x))
    axs[1].legend()
    
    axs[2].set_ylim(0, max(x))
    axs[2].legend()
   
    x = envs[1].reshape((-1, ))
    y = envs[2].reshape((-1, ))
    ax.scatter(x, y, s=2, alpha=0.1)
    ax.set_xlabel('S')
    ax.set_ylabel('V')

    print('| H mean =', np.mean(envs[0]))
    print('| S mean =', np.mean(envs[1]))
    print('| V mean =', np.mean(envs[2]))

    plt.show()