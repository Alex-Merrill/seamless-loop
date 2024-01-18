import multiprocessing
import os
import shutil
import time
from functools import wraps

import colour
import cv2 as cv
import numpy as np


# TODO: create heuristic that combines color diff + optical flow
# to calc best frame


def timeit(action, include_id=False, end_line=False):
    """
    decorator to print time taken to run function

    takes an action input that describes task of function being done
    """

    def wrap(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            endline = "\n" if end_line else ""
            id = f" {args[1]}" if include_id else ""
            print(f"{action}{id}")
            start_time = time.perf_counter()
            res = f(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            print(
                f"Done {action[0].lower()}{action[1:]}{id}!"
                f" Took {round(elapsed, 4)} seconds{endline}"
            )
            return res

        return wrapped_f

    return wrap


class Looper:
    def __init__(self, min_duration, max_duration, gray, color_diff, src):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.src = src
        self.gray = gray
        self.color_diff = color_diff
        self.DOWNSAMPLE_FACTOR = 0.25
        self.get_frame_pixel_diff = None
        if self.gray:
            self.get_frame_pixel_diff = self.get_frame_pixel_diff_gray
        else:
            if not self.color_diff == "CIE 2000":
                self.get_frame_pixel_diff = self.get_frame_pixel_diff_color_CIE
            else:
                self.get_frame_pixel_diff = self.get_frame_pixel_diff_color_redmean

        print(
            "Initializing loop calculations...\n"
            f"Minimum loop duration: {self.min_duration}\n"
            f"Maximum loop duration: {self.max_duration}\n"
            f"Grayscale mode: {'Active' if self.gray else 'Disabled'}\n"
            f"Color difference method: {'Basic diff' if self.gray else 'Redmean' if self.color_diff == 'redmean' else 'CIE 2000'}\n"
            f"Source file: {self.src}\n"
        )

    def Start(self):
        self.read_video()
        cv.destroyAllWindows()
        self.get_all_frame_diffs()
        self.get_best_loop()
        self.write_best_loop()
        return

    @timeit("Reading video file", end_line=True)
    def read_video(self):
        """
        reads video frames and stores each color frame in self.frame_buf_c
        and each gray scale frame in self.frame_buf_g as well as calculates
        optical flow and stores in self.flow_buf. Also saves some data about
        video that is needed later. All frames are downsampled by a factor
        of .25 using cubic interpolation.
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
        prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

        self.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.frame_width = prev.shape[1]
        self.frame_height = prev.shape[0]
        print(
            "Video Data:\n"
            f"Frame count: {self.frame_count}\n"
            f"FPS: {self.fps}\n"
            f"Frame Width: {self.frame_width}\n"
            f"Frame Height: {self.frame_height}"
        )

        self.frame_buf_c = np.empty(
            (self.frame_count, *prev.shape), np.dtype("float32")
        )
        self.frame_buf_g = np.empty(
            (self.frame_count, *prev_gray.shape), np.dtype("float32")
        )
        self.flow_buf = np.empty(
            (self.frame_count, self.frame_height, self.frame_width, 2),
            np.dtype("float32"),
        )

        self.frame_buf_c[fc] = prev
        self.frame_buf_g[fc] = prev_gray
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
            curr = cv.cvtColor(curr, cv.COLOR_BGR2RGB)
            curr_gray = cv.cvtColor(curr, cv.COLOR_RGB2GRAY)
            self.frame_buf_c[fc] = curr
            self.frame_buf_g[fc] = curr_gray

            # calc optical flow
            flow = cv.calcOpticalFlowFarneback(
                self.frame_buf_g[fc - 1],
                self.frame_buf_g[fc],
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            self.flow_buf[fc] = flow
            fc += 1

            # cv.imshow("flow", self.draw_flow(gray, flow))

            ch = cv.waitKey(5)
            if ch == 27:
                break

        cap.release()

    @timeit("Calculating best loop frames")
    def get_best_loop(self):
        best_per_fc = [(float("infinity"), -1, -1)] * (self.fps * self.max_duration + 1)
        for f1, f1_diffs in enumerate(self.frame_diffs):
            for f2, diff in enumerate(f1_diffs):
                if not diff:
                    continue
                fc = f2 - f1 + 1
                if diff < best_per_fc[fc][0]:
                    best_per_fc[fc] = (diff, f1, f2)

        best_overall = (float("infinity"), -1, -1)
        for i, diff in enumerate(best_per_fc):
            if diff[0] != float("infinity"):
                print(f"fc: {i} | diff: {diff}")
                if diff[0] < best_overall[0]:
                    best_overall = diff

        print(f"best overall: {best_overall}")

        self.best_start = best_overall[1]
        self.best_end = best_overall[2]

    def write_best_loop(self):
        """
        writes gif to project dir of best loop found
        """

        # clean up any previous tmp dir and gif
        tmp_dir = "./tmp_images"
        if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        if os.path.exists("./output.gif"):
            os.remove("./output.gif")
        os.makedirs(tmp_dir)

        # writes frames to tmp dir and saves paths
        # TODO: just read from actual video to get full res frames
        image_paths = []
        for i in range(self.best_start, self.best_end + 1):
            output_filename = str(i)
            while len(output_filename) < 3:
                output_filename = "0" + output_filename
            output_filename += ".png"
            im_path = os.path.join(tmp_dir, output_filename)
            image_paths.append(im_path)
            cv.imwrite(im_path, self.frame_buf_c[i])

        # use imagemagick to create gif from frames
        delay = 1 / self.fps
        loop = 0
        output_path = "./output.gif"
        cmd = (
            f"convert -delay {delay} {' '.join(image_paths)} -loop {loop} {output_path}"
        )
        os.system(cmd)

        # cleanup tmp dir
        if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    @timeit("Getting all frame pixel diffs")
    def get_all_frame_diffs(self):
        """
        calculates all frame pixel differences for valid frames to be
        considered in the loop

        returns array where frame_diffs[i] = [avg_diff(i,0),...,avg_diff(i,n)]
        """
        self.frame_diffs = []

        last_frame_eligible = self.frame_count - (self.fps * self.min_duration)
        pool = multiprocessing.Pool()
        res = pool.map(self.get_frame_diffs, list(range(last_frame_eligible + 1)))

        for diff in res:
            self.frame_diffs.append(diff)

    @timeit(
        "Getting pixel diffs for frame",
        include_id=True,
    )
    def get_frame_diffs(self, idx):
        """
        calculates average redmean pixel difference for frame idx and every
        other valid frame considering the min and max loop duration

        returns an array where diffs[i] = avg pixel difference between
        frame[idx] and frame[i]
        """
        diffs = [-1] * self.frame_count

        first_frame_eligible = idx + (self.fps * self.min_duration) - 1
        last_frame_eligible = idx + (self.fps * self.max_duration) - 1
        last_frame_eligible = min(last_frame_eligible, self.frame_count - 1)
        for i in range(first_frame_eligible, last_frame_eligible + 1):
            diffs[i] = self.get_frame_pixel_diff(idx, i)

        return diffs

    def get_frame_pixel_diff_color_redmean(self, f1_idx, f2_idx):
        """
        calculates redmean pixel difference for each pixel in frames f1 and f2

        returns average redmean difference for all pixels in frame
        """
        f1, f2 = self.frame_buf_c[f1_idx], self.frame_buf_c[f2_idx]
        r_bar = (f1[..., 0] + f2[..., 0]) / 2
        d_r = f1[..., 0] - f2[..., 0]
        d_g = f1[..., 1] - f2[..., 1]
        d_b = f1[..., 2] - f2[..., 2]

        p1 = (2 + r_bar / 256) * np.square(d_r)
        p2 = np.square(d_g) * 4
        p3 = (2 + (255 - r_bar) / 256) * np.square(d_b)
        diff = np.sqrt(p1 + p2 + p3)

        return np.mean(diff)

    def get_frame_pixel_diff_color_CIE(self, f1_idx, f2_idx):
        """
        calculates CIE 2000 color difference between two frames

        returns average difference for all pixels in frame
        """
        f1, f2 = self.frame_buf_c[f1_idx], self.frame_buf_c[f2_idx]
        f1_lab = cv.cvtColor(f1, cv.COLOR_RGB2Lab)
        f2_lab = cv.cvtColor(f2, cv.COLOR_RGB2Lab)
        delta_e = colour.delta_E(f1_lab, f2_lab, method="CIE 2000")
        return np.mean(delta_e)

    def get_frame_pixel_diff_gray(self, f1_idx, f2_idx):
        """
        calculates grayscale pixel difference for each pixel in frames f1 and f2

        returns average grayscale difference for all pixels in frame
        """
        f1, f2 = self.frame_buf_g[f1_idx], self.frame_buf_g[f2_idx]
        diff = np.absolute(f1 - f2)
        return np.mean(diff)

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
