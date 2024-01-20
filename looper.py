import os
import subprocess
import shutil
import sys
import traceback
import time
from functools import wraps

from numpy.lib import math

from delta_e import delta_E_CIE2000
import cv2 as cv
import numpy as np


# TODO: create heuristic that combines color diff + optical flow
# to calc best frame


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def print_exception(src):
    e_type, value, tb = sys.exc_info()
    info, error = traceback.format_exception(e_type, value, tb)[-2:]
    error = error.strip("\n")
    print(f"{src}: {bcolors.FAIL}{error}")
    print(f"[Traceback]:\n {bcolors.FAIL}{info}")


def timeit(
    action,
    ansi="",
    include_id=False,
    end_new_line=False,
    start_new_line=False,
):
    """
    decorator to print time taken to run function

    :param action: describes function action
    :type action: string
    :param include_id: whether to inlude id param of function (first param)
    :type include_id: bool
    :param end_new_line: whether to print a new line after elapsed time
    :type end_new_line: bool
    :param start_new_line: whether to print a new line after action
    :type start_new_line: bool
    """

    def wrap(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            startline = "\n" if start_new_line else ""
            endline = "\n" if end_new_line else ""
            id = f" {args[1]}" if include_id else ""
            print(f"[INFO] {ansi}{action}{id}{startline}{bcolors.RESET}")
            start_time = time.perf_counter()
            res = f(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            print(
                f"[INFO] {ansi}Done {action[0].lower()}"
                f"{action[1:]}{id}!{bcolors.RESET} Took {round(elapsed, 4)} "
                f"seconds{endline}"
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
        self.corrupted = False
        if self.gray:
            self.get_frame_pixel_diff = self.get_frame_pixel_diff_gray
        else:
            if self.color_diff == "CIE 2000":
                self.get_frame_pixel_diff = self.get_frame_pixel_diff_color_CIE
            else:
                self.get_frame_pixel_diff = self.get_frame_pixel_diff_color_redmean

        print(
            f"[INFO] {bcolors.BOLD}{bcolors.HEADER}Initializing loop calculations...\n{bcolors.RESET}"
            f"[INFO] {bcolors.OKGREEN}Minimum loop duration: {self.min_duration}\n{bcolors.RESET}"
            f"[INFO] {bcolors.OKGREEN}Maximum loop duration: {self.max_duration}\n{bcolors.RESET}"
            f"[INFO] {bcolors.OKGREEN}Grayscale mode: {'Active' if self.gray else 'Disabled'}\n{bcolors.RESET}"
            f"[INFO] {bcolors.OKGREEN}Color difference method: {'Basic diff' if self.gray else 'Redmean' if self.color_diff == 'redmean' else 'CIE 2000'}\n{bcolors.RESET}"
            f"[INFO] {bcolors.OKGREEN}Source file: {self.src}\n{bcolors.RESET}"
        )

    @timeit(
        "Finding loop",
        start_new_line=True,
        ansi=f"{bcolors.BOLD}{bcolors.UNDERLINE}{bcolors.HEADER}",
    )
    def Start(self):
        """starts looper on video to get best start/end frames and write gif"""
        try:
            self.read_video()
            self.get_all_frame_diffs()
            self.get_best_loop()
            self.write_best_loop()
        except cv.error:
            print_exception("[OpenCV Error]")
        except Exception:
            print_exception("[Error]")

    @timeit(
        "Reading video file", ansi=f"{bcolors.BOLD}{bcolors.HEADER}", end_new_line=True
    )
    def read_video(self):
        """
        reads video frames and stores each color frame in self.frame_buf_c
        and each gray scale frame in self.frame_buf_g as well as calculates
        optical flow and stores in self.flow_buf. Also saves some data about
        video that is needed later. All frames are downsampled by a factor
        of .25 using cubic interpolation.
        """
        try:
            cap = cv.VideoCapture(self.src)

            # read first frame for optical flow calcs + frame size/count
            fc = 0
            ret, frame = cap.read()

            if not ret:
                raise Exception("Video file cannot be read")

            frame = cv.resize(
                frame,
                None,
                fx=self.DOWNSAMPLE_FACTOR,
                fy=self.DOWNSAMPLE_FACTOR,
                interpolation=cv.INTER_CUBIC,
            )
            prev = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            prev_gray = cv.cvtColor(prev, cv.COLOR_RGB2GRAY)

            supposed_frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            supposed_fps = int(cap.get(cv.CAP_PROP_FPS))

            frame_buf_c = [prev]
            frame_buf_g = [prev_gray]
            flow_buf = []
            fc = 1

            while True:
                ret, frame = cap.read()
                if not ret:
                    if fc != supposed_frame_count:
                        self.corrupted = True
                    break

                frame = cv.resize(
                    frame,
                    None,
                    fx=self.DOWNSAMPLE_FACTOR,
                    fy=self.DOWNSAMPLE_FACTOR,
                    interpolation=cv.INTER_CUBIC,
                )
                curr = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                curr_gray = cv.cvtColor(curr, cv.COLOR_RGB2GRAY)
                frame_buf_c.append(curr)
                frame_buf_g.append(curr_gray)

                # calc optical flow
                flow = cv.calcOpticalFlowFarneback(
                    frame_buf_g[fc - 1],
                    frame_buf_g[fc],
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
                flow_buf.append(flow)
                fc += 1

                # cv.imshow("flow", self.draw_flow(gray, flow))

                ch = cv.waitKey(5)
                if ch == 27:
                    break

            cap.release()
            # cv.destroyAllWindows()

            if self.corrupted:
                self.frame_count = fc
                get_duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {self.src}"
                ffprobe_out = subprocess.run(
                    get_duration_cmd, capture_output=True, shell=True, text=True
                )
                if ffprobe_out.returncode == 1:
                    raise Exception("Not able to read duration of video using ffprobe")
                duration = float(ffprobe_out.stdout)
                self.fps = fc / duration
                self.duration = duration
            else:
                self.frame_count = supposed_frame_count
                self.fps = supposed_fps

            self.frame_width = frame_buf_c[0].shape[1]
            self.frame_height = frame_buf_c[0].shape[0]

            loop_dur_invalid = False
            if self.duration < self.max_duration:
                loop_dur_invalid = True
                self.max_duration = self.duration
            if self.duration < self.min_duration:
                self.min_duration = self.duration / 2

            if self.corrupted:
                print(
                    f"[WARNING] {bcolors.WARNING}Video Corrupted! Will attempt to make things work.{bcolors.RESET}"
                )

            if loop_dur_invalid:
                print(
                    f"[WARNING] {bcolors.WARNING}Specified loop durations invalid!\n{bcolors.RESET}"
                    f"[WARNING] {bcolors.WARNING}Adjusting loop durations to:\n{bcolors.RESET}"
                    f"[WARNING] {bcolors.WARNING}Minimum loop duration: {self.min_duration}s\n{bcolors.RESET}"
                    f"[WARNING] {bcolors.WARNING}Maximum loop duration: {self.max_duration}s{bcolors.RESET}"
                )
            print(f"[INFO] {bcolors.OKBLUE}Video Data{bcolors.RESET}")
            print(
                f"[INFO] Frame count: {self.frame_count}\n"
                f"[INFO] FPS: {self.fps}\n"
                f"[INFO] Duration: {self.duration}\n"
                f"[INFO] Frame Width: {self.frame_width}\n"
                f"[INFO] Frame Height: {self.frame_height}"
            )

            self.frame_buf_c = np.array(frame_buf_c).astype(np.float32)
            self.frame_buf_g = np.array(frame_buf_g).astype(np.float32)
            self.flow_buf = np.array(flow_buf).astype(np.float32)

        except cv.error as e:
            raise e
        except Exception as e:
            raise e

    @timeit(
        "Calculating best loop frames",
        ansi=f"{bcolors.BOLD}{bcolors.HEADER}",
        end_new_line=True,
    )
    def get_best_loop(self):
        best_per_fc = [(float("infinity"), -1, -1)] * (
            math.ceil(self.fps * self.max_duration) + 1
        )
        for f1, f1_diffs in enumerate(self.frame_diffs):
            for f2, diff in enumerate(f1_diffs):
                if diff == -1:
                    continue
                fc = f2 - f1 + 1
                if diff < best_per_fc[fc][0]:
                    best_per_fc[fc] = (diff, f1, f2)

        best_overall = (float("infinity"), -1, -1)
        for i, diff in enumerate(best_per_fc):
            if diff[0] != float("infinity"):
                print(f"[INFO] fc: {i} | diff: {diff}")
                if diff[0] < best_overall[0]:
                    best_overall = diff

        print(f"[INFO] best overall: {best_overall}")

        self.best_start = best_overall[1]
        self.best_end = best_overall[2]

    @timeit(
        "Writing frames to gif",
        ansi=f"{bcolors.BOLD}{bcolors.HEADER}",
        end_new_line=True,
    )
    def write_best_loop(self):
        """
        writes gif to project dir of best loop found
        """

        # clean up any previous tmp dir and gif
        tmp_dir = "./tmp_images"
        if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        # writes frames to tmp dir and saves paths
        cap = cv.VideoCapture(self.src)
        fc = 0
        image_paths = []
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if fc >= self.best_start and fc <= self.best_end:
                output_filename = f"0{fc}.png"
                im_path = os.path.join(tmp_dir, output_filename)
                print(f"[INFO] writing frame ({fc}) to path ({im_path})")
                image_paths.append(im_path)
                cv.imwrite(im_path, frame)

            fc += 1

        # use imagemagick to create gif from frames
        delay = 1 / self.fps
        loop = 0
        output_name_base = "output"
        output_name = output_name_base
        i = 1
        while os.path.exists(f"./{output_name}.gif"):
            output_name = f"{output_name_base}{i}"
            i += 1
        output_path = f"./{output_name}.gif"
        cmd = (
            f"convert -delay {delay} {' '.join(image_paths)} -loop {loop} {output_path}"
        )
        os.system(cmd)

        # cleanup tmp dir
        if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    @timeit(
        "Getting all frame pixel diffs",
        ansi=f"{bcolors.BOLD}{bcolors.HEADER}",
        end_new_line=True,
    )
    def get_all_frame_diffs(self):
        """
        calculates all frame pixel differences for valid frames to be
        considered in the loop

        returns array where frame_diffs[i] = [avg_diff(i,0),...,avg_diff(i,n)]
        """
        self.frame_diffs = []
        last_frame_eligible = self.frame_count - math.ceil(self.fps * self.min_duration)
        for i in range(last_frame_eligible + 1):
            self.frame_diffs.append(self.get_frame_diffs(i))

    @timeit(
        "Getting pixel diffs for frame",
        # ansi=f"{bcolors.BOLD}{bcolors.HEADER}",
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

        first_frame_eligible = idx + math.ceil(self.fps * self.min_duration) - 1
        last_frame_eligible = idx + math.ceil(self.fps * self.max_duration) - 1
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
        delta_e = delta_E_CIE2000(f1_lab, f2_lab)
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
