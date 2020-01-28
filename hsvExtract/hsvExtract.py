import cv2
import numpy
import matplotlib.pyplot as plt


def getImage(path):
    return  cv2.imread(cv2.samples.findFile(path))

if __name__ == "__main__":

    orange = getImage("orange.png")
    orange_dark  = getImage("orange_dark.png")
    orange_light = getImage("orange_light.png")

    # cv2.imshow("orange", orange)
    # cv2.imshow("orange_dark", orange_dark)
    # cv2.imshow("orange_light", orange_light)
    orangeConcat = cv2.hconcat([orange_dark, orange, orange_light])
    cv2.imshow("orange", orangeConcat)

    h, w, c = orange.shape

    print("| orange")
    print("| >> \t shape =", orange.shape)

    print("| orange_dark")
    print("| >> \t shape =", orange_dark.shape)
    
    print("| orange_light")
    print("| >> \t shape =", orange_light.shape)

    orange_hsv = cv2.cvtColor(orange, cv2.COLOR_BGR2HSV)
    orange_dark_hsv = cv2.cvtColor(orange_dark, cv2.COLOR_BGR2HSV)
    orange_light_hsv = cv2.cvtColor(orange_light, cv2.COLOR_BGR2HSV)

    oranges = cv2.split(orange_hsv)
    oranges_dark = cv2.split(orange_dark_hsv)
    oranges_light = cv2.split(orange_light_hsv)

    fig, axs = plt.subplots(c, 1, figsize=(8, 8))

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(c):

        x = oranges[i].reshape((-1, ))
        axs[i].hist(x, color='blue', bins=40, alpha=0.3, normed=True, label='orange')

        x = oranges_dark[i].reshape((-1, ))
        axs[i].hist(x, color='green', bins=40, alpha=0.3, normed=True, label='orange_dark')
        
        x = oranges_light[i].reshape((-1, ))
        axs[i].hist(x, color='red', bins=40, alpha=0.3, normed=True, label='orange_light')

        axs[i].set_ylim(0, 1)
        axs[i].legend()

    orangesArrays = [oranges, oranges_dark, oranges_light]
    name = ['orange', 'orange_dark', 'orange_light']

    for i in range(3):
        orangesArray = orangesArrays[i]
        x = orangesArray[1].reshape((-1, ))
        y = orangesArray[2].reshape((-1, ))
        ax.scatter(x, y, s=2, alpha=0.1, label=name[i])

    ax.legend()
    plt.show()