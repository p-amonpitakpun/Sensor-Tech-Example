import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def getImage(path):
    return  cv2.imread(cv2.samples.findFile(path))

def hist(x):
    xHist, xEdge = np.histogram(x)
    xCenter = (xEdge[:-1] + xEdge[1:]) / 2
    return xHist, xCenter

if __name__ == '__main__':
    path_list = glob.glob("color\\*.jpg")
    image_dict = {}
    for path in path_list:
        img = getImage(path)
        img = cv2.resize(img, (200, 200))
        image_dict[path] = img
    # N = len(image_dict)
    # n_row = int(np.sqrt(N))
    # n_col = int(np.ceil(N / n_row))

    # k = list(image_dict.keys())[0]
    # square = image_dict[k]
    # cv2.imshow(k, square)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # fig2, ax2 = plt.subplots(figsize=(10, 10))
    # fig3, ax3 = plt.subplots(figsize=(10, 10))

    # axs[0].set_xscale('log')
    # axs[1].set_xscale('log')

    for path, img in image_dict.items():

        name = path[path.find("\\") + 1 : path.find(".jpg")]

        channel = cv2.split(img)

        b = np.array(channel[0].flatten(), dtype=float)
        g = np.array(channel[1].flatten(), dtype=float)
        r = np.array(channel[2].flatten(), dtype=float)

        color = (np.average(r) / 255, np.average(g) / 255, np.average(b) / 255, 1)

        rgb = r + g + b

        r /= rgb
        g /= rgb
        b /= rgb

        axs[0][0].scatter(g, r, color=color, s=1, label=name)
        axs[0][1].scatter(g, b, color=color, s=1, label=name)
        axs[1][0].scatter(b, r, color=color, s=1, label=name)

        print("{")
        print("\t\""+name+"\",")
        print("\t{", "{:.9f} , {:.9f} , {:.9f}".format(color[0], color[1], color[2]), "}")
        print("},")

        # rg = r / g
        # bg = b / g

        # rgMean = int(np.mean(rg) / 1e-3) * 1e-3
        # rgVar = int(np.var(rg) / 1e-6) * 1e-6
        # bgMean = int(np.mean(bg) / 1e-3) * 1e-3
        # bgVar = int(np.var(bg) / 1e-6) * 1e-6

        # print('rg', rgMean, rgVar)
        # print('bg', bgMean, bgVar)

        # color = (np.average(r) / 255, np.average(g) / 255, np.average(b) / 255, 1)
        
        # rgHist, rgCenter = hist((r / g)[np.isfinite(r / g)])
        # bgHist, bgCenter = hist((b / g)[np.isfinite(b / g)])

        # axs[0].plot(rgCenter, rgHist, c=color)
        # axs[1].plot(bgCenter, bgHist, c=color)

        ## Ratio of G

        # s = path + str((rgMean, rgVar, bgMean, bgVar))
        # ax2.scatter(bg, rg, color=color, s=1, alpha=0.1, label=s)
        # ax2.set_xlabel('B / G')
        # ax2.set_ylabel('R / G')

        ## Normalize

        # rgb = r + g + b

        # plt.scatter(b, r)

        # print(max(r / rgb), min(r / rgb))

        # s = path + str((np.mean(r) / np.mean(rgb), np.mean(b) / np.mean(rgb)))
        # ax3.scatter(b / rgb, r / rgb, color=color, s=1, alpha=0.01)
        # ax3.set_xlabel('B / RGB')
        # ax3.set_ylabel('R / RGB')
        # ax3.set_xlim(0, 1)
        # ax3.set_ylim(0, 1)

        # pair = np.array([b / rgb, r / rgb]).T
        # rbpair = []
        # for p in pair:
        #     if np.isfinite(p).all():
        #         rbpair.append(p)
        # rbpair = np.array(rbpair)

        # cov = np.cov(rbpair.T).reshape((-1, ))
        # bR = np.mean((b / rgb)[np.isfinite(b / rgb)])
        # rR = np.mean((r / rgb)[np.isfinite(r / rgb)])
        # print("{", bR, ',', rR,',', cov[0], ",", cov[1], ",", cov[2], ",", cov[3],"}, ")

    plt.legend()
    # fig.savefig('hist.png')
    # fig2.savefig('scatter.png')
    plt.show()
