import math
import sys
import time

import cv2
import numpy as np
from PIL import Image


def readVideo(path):
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype("float"))

    fc = 0
    ret = True
    while fc < frameCount and ret:
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    # cv2.imshow('frame 10', buf[9])
    # cv2.waitKey()

    return buf


def getFrameDiffs(frames, idx):
    diffs = [0] * len(frames)
    for i in range(len(frames)):
        print(f"frame {i}")
        diffs[i] = getFrameDiff(frames[idx], frames[i])

    return diffs


def getFrameDiff(f1, f2):
    totalDiff = 0.0

    for row in range(len(f1)):
        for col in range(len(f1[row])):
            rgb1 = f1[row][col]
            rgb2 = f2[row][col]
            r1 = rgb1[0]
            g1 = rgb1[1]
            b1 = rgb1[2]
            r2 = rgb2[0]
            g2 = rgb2[1]
            b2 = rgb2[2]

            rBar = (r1 + r2) / 2
            dR = r1 - r2
            dB = b1 - b2
            dG = g1 - g2

            p1 = (2 + rBar / 256) * dR**2
            p2 = 4 * dG**2
            p3 = (2 + (255 - rBar) / 256) * dB**2

            stuff = p1 + p2 + p3
            dC = math.sqrt(stuff)

            totalDiff += dC

            totalDiff /= len(f1)

    return totalDiff


if __name__ == "__main__":
    path = sys.argv[1]
    startTime = time.time()
    frames = readVideo(path)
    endTime = time.time()
    print(endTime - startTime)
    diffs = getFrameDiffs(frames, 0)
    print(diffs)
