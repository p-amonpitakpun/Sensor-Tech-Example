import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import threading
import time

class Data:

    def __init__(self):
        self.x = []
        self.y = []

if __name__ == '__main__':
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='polar') ## , projection='polar'
    ax.set_ylim(0,20)

    n = 100

    dat = Data()
    l  = ax.scatter([],[], c='r', s=1)

    def update(i):
        global dat
        dat.y = np.random.rand(n)*5 + 10
        dat.x = np.linspace(0,2.*np.pi, num=n)
        dat.y[-1] = dat.y[0]
        plt.title(i)
        l.set_offsets(np.append(dat.x.reshape((-1, 1)), dat.y.reshape((-1, 1)), axis=1))
        return l, 

    ani = animation.FuncAnimation(fig, update, frames=50, interval=200, blit=True)
    plt.show()