import math
import sys
import time

import cv2 as cv
import numpy as np


class Looper:
    def __init__(self, min_duration, max_duration, src):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.src = src

    def Start(self):
        self.read_video()
        cv.destroyAllWindows()
        return

    def read_video(self):
        cap = cv.VideoCapture(self.src)

        fc = 0
        ret, frame = cap.read()
        prev = cv.resize(
            frame,
            None,
            fx=0.4,
            fy=0.4,
            interpolation=cv.INTER_CUBIC,
        )
        prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        self.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.frame_width = prev.shape[1]
        self.frame_height = prev.shape[0]
        self.frame_buf = np.empty(
            (self.frame_count, self.frame_height, self.frame_width, 3),
            np.dtype("uint8"),
        )
        self.frame_buf[fc] = prev
        self.flow_buf = np.empty(
            (self.frame_count, self.frame_height, self.frame_width, 2),
            np.dtype("uint8"),
        )
        fc += 1

        while fc < self.frame_count and ret:
            ret, frame = cap.read()
            curr = cv.resize(
                frame,
                None,
                fx=0.4,
                fy=0.4,
                interpolation=cv.INTER_CUBIC,
            )
            self.frame_buf[fc] = curr
            gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(
                prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            self.flow_buf[fc] = flow
            prevgray = gray
            fc += 1

            cv.imshow("flow", self.draw_flow(gray, flow))

            ch = cv.waitKey(5)
            if ch == 27:
                break

        cap.release()

    def draw_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = (
            np.mgrid[step / 2 : h : step, step / 2 : w : step]
            .reshape(2, -1)
            .astype(int)
        )
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis
