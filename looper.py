import math
import sys
import time
from functools import wraps

import cv2 as cv
import numpy as np


def timeit(action, finish, include_id=False):
    """
    decorator to print time taken to run function

    takes an action input that describes task of function being done
    """

    def wrap(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            test = f" {args[1]}" if include_id else ""
            test = args[1] if include_id else ""
            print(f"{action} {test}")
            start_time = time.time()
            res = f(*args, **kwargs)
            end_time = time.time()
            print(f"Done {finish}{test}! Took {end_time - start_time} seconds")
            return res

        return wrapped_f

    return wrap


class Looper:
    def __init__(self, min_duration, max_duration, src):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.src = src
        self.DOWNSAMPLE_FACTOR = 0.25

    def Start(self):
        self.read_video()
        cv.destroyAllWindows()
        self.getAllFrameDiffs()
        return

    @timeit("Reading video file...", "reading video file")
    def read_video(self):
        """
        reads video frame by frame and stores each frame in self.frame_buf
        as well as calculates optical flow and stores in self.flow_buf. Also
        saves some data about video that is needed later. All frames are
        downsampled by a factor of .4 using cubic interpolation.
        """

        cap = cv.VideoCapture(self.src)

        # read first frame for optical flow calcs + frame size/count
        fc = 0
        ret, frame = cap.read()
        prev = cv.resize(
            frame,
            None,
            fx=self.DOWNSAMPLE_FACTOR,
            fy=self.DOWNSAMPLE_FACTOR,
            interpolation=cv.INTER_CUBIC,
        )
        prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        self.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.frame_width = prev.shape[1]
        self.frame_height = prev.shape[0]
        self.frame_buf = np.empty(
            (self.frame_count, self.frame_height, self.frame_width, 3),
            np.dtype("float"),
        )
        self.frame_buf[fc] = prev
        self.flow_buf = np.empty(
            (self.frame_count, self.frame_height, self.frame_width, 2),
            np.dtype("float"),
        )
        fc += 1

        while fc < self.frame_count and ret:
            ret, frame = cap.read()
            curr = cv.resize(
                frame,
                None,
                fx=self.DOWNSAMPLE_FACTOR,
                fy=self.DOWNSAMPLE_FACTOR,
                interpolation=cv.INTER_CUBIC,
            )
            self.frame_buf[fc] = curr

            # convert to graysale and calc optical flow
            gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(
                prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            self.flow_buf[fc] = flow
            prevgray = gray
            fc += 1

            # cv.imshow("flow", self.draw_flow(gray, flow))

            ch = cv.waitKey(5)
            if ch == 27:
                break

        cap.release()

    @timeit("Getting all frame pixel diffs...", "getting all frame pixel diffs")
    def getAllFrameDiffs(self):
        frame_diffs = np.empty(
            (self.frame_count, self.frame_count), np.dtype("float")
        )

        last_frame_eligible = self.frame_count - (self.fps * self.min_duration)
        for i in range(last_frame_eligible):
            frame_diffs[i] = self.getFrameDiffs(i)

        return frame_diffs

    @timeit(
        "Getting pixel diffs for frame",
        "getting pixel diffs for frame",
        include_id=True,
    )
    def getFrameDiffs(self, idx):
        diffs = [0] * len(self.frame_buf)
        for i in range(idx + 1, len(self.frame_buf)):
            diffs[i] = self.getFramePixelDiff(
                self.frame_buf[idx], self.frame_buf[i]
            )

        return diffs

    def getFramePixelDiff(self, f1, f2):
        """
        calculates redmean pixel difference for each pixel in frames f1 and f2

        returns average redmean difference for all pixels in frame
        """
        total_diff = 0.0
        for row in range(len(f1)):
            for col in range(len(f1[0])):
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

                squared_diff = p1 + p2 + p3
                total_diff += squared_diff

        total_diff /= len(f1) * len(f1[0])

        return total_diff

    def draw_flow(self, img, flow, step=16):
        """
        draws flow vectors on input img

        takes frame data and flow data
        """
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
