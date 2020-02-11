import cv2
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="file parser")
    parser.add_argument('filename')
    return parser.parse_args(args)


def getImage(path):
    return  cv2.imread(cv2.samples.findFile(path))

if __name__ == '__main__':
    args = parse()
    filename = args.filename
    print(filename)
    image = getImage(filename)
    cv2.imshow(filename, image)

    env = image
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
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)

    print('| H mean =', np.mean(envs[0]))
    print('| S mean =', np.mean(envs[1]))
    print('| V mean =', np.mean(envs[2]))

    plt.show()
