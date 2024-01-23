import os
import subprocess
import shutil
import sys
import traceback
import time
from functools import wraps
import math
import cv2 as cv
import numpy as np
from typing import Any, Literal

from delta_e import delta_E_CIE2000


# TODO: pull flow_diff coefficients out to user input
# TODO: add user input on whether to use optical flow?
# TODO: handle noise in both color difference and optical flow difference
# TODO: maybe think about some other heurisitc function for noise difference
# TODO: maybe use nptyping for typing shape of numpy array?


class pfmt:
    HEADER = "\033[95m"
    BHEADER = "\033[1m\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def print_exception():
    e_type, value, tb = sys.exc_info()
    info, error = traceback.format_exception(e_type, value, tb)[-2:]
    error = error.strip("\n")
    print(f"[ERROR]: {pfmt.FAIL}{error}{pfmt.RESET}")
    print(f"[Traceback]:\n {pfmt.FAIL}{info}{pfmt.RESET}")


def timeit(
    action,
    ansi="",
    include_id=False,
    end_new_line=False,
    start_new_line=False,
):
    """
    decorator to log progress and track time

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
            print(f"[INFO] {ansi}{action}{id}{startline}{pfmt.RESET}")
            start_time = time.perf_counter()
            res = f(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            print(
                f"[INFO] {ansi}Done {action[0].lower()}"
                f"{action[1:]}{id}!{pfmt.RESET} Took {round(elapsed, 4)} "
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
        diff_method = "gray" if self.gray else self.color_diff
        if diff_method == "gray":
            self.get_color_diff = self.get_frame_color_diff_gray
        elif diff_method == "CIE 2000":
            self.get_color_diff = self.get_frame_color_diff_CIE_2000
        elif diff_method == "redmean":
            self.get_color_diff = self.get_frame_color_diff_redmean
        else:
            raise Exception(
                f"Color difference method unkown: {diff_method}. "
                "This shouldn't happen..."
            )

        print(
            f"[INFO] {pfmt.BHEADER}Initializing loop calculations..."
            f"\n{pfmt.RESET}"
            f"[INFO] {pfmt.OKGREEN}Minimum loop duration: {self.min_duration}"
            f"\n{pfmt.RESET}"
            f"[INFO] {pfmt.OKGREEN}Maximum loop duration: {self.max_duration}"
            f"\n{pfmt.RESET}"
            f"[INFO] {pfmt.OKGREEN}Color difference method: {diff_method}"
            f"\n{pfmt.RESET}"
            f"[INFO] {pfmt.OKGREEN}Source file: {self.src}\n{pfmt.RESET}"
        )

    @timeit(
        "Finding loop",
        start_new_line=True,
        ansi=f"{pfmt.BHEADER}{pfmt.UNDERLINE}",
    )
    def Start(self):
        """starts looper on video to get best start/end frames and write gif"""
        try:
            self.read_video()
            self.get_all_frames_color_diffs()
            self.get_all_frames_flow_conts()
            # self.get_best_loop_color_flow()
            self.get_best_loop_color()
            self.write_best_loop_frames()
            self.create_gif()
        except Exception:
            print_exception()
            sys.exit(1)

    @timeit("Reading video file", ansi=f"{pfmt.BHEADER}", end_new_line=True)
    def read_video(self):
        """
        reads video frames and stores each color frame in self.frame_buf_c
        and each gray scale frame in self.frame_buf_g as well as calculates
        optical flow and stores in self.flow_buf. Also saves some data about
        video that is needed later. All frames are downsampled by a factor
        of self.DOWNSAMPLE_FACTOR using cubic interpolation.
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
                flow = np.empty((*curr_gray.shape[:2], 2), dtype=np.float32)
                cv.calcOpticalFlowFarneback(
                    frame_buf_g[fc - 1],
                    frame_buf_g[fc],
                    flow,
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

            cap.release()

            if self.corrupted:
                self.frame_count = fc
                get_duration_cmd = (
                    "ffprobe -v error -show_entries "
                    f"format=duration -of default=noprint_wrappers=1:nokey=1 "
                    f"{self.src}"
                )
                ffprobe_out = subprocess.run(
                    get_duration_cmd,
                    capture_output=True,
                    shell=True,
                    text=True,
                )
                if ffprobe_out.returncode == 1:
                    raise Exception(
                        "Unable to read duration of video using ffprobe"
                    )
                duration = float(ffprobe_out.stdout)
                self.fps = fc / duration
                self.duration = duration
            else:
                self.frame_count = supposed_frame_count
                self.fps = supposed_fps
                self.duration = self.frame_count / self.fps

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
                    f"[WARNING] {pfmt.WARNING}Video Corrupted! Will attempt "
                    f"to make things work.{pfmt.RESET}"
                )

            if loop_dur_invalid:
                print(
                    f"[WARNING] {pfmt.WARNING}Specified loop durations "
                    f"invalid!\n{pfmt.RESET}"
                    f"[WARNING] {pfmt.WARNING}Adjusting loop durations to:"
                    f"\n{pfmt.RESET}"
                    f"[WARNING] {pfmt.WARNING}Minimum loop duration: "
                    f"{self.min_duration}s\n{pfmt.RESET}"
                    f"[WARNING] {pfmt.WARNING}Maximum loop duration: "
                    f"{self.max_duration}s{pfmt.RESET}"
                )
            print(f"[INFO] {pfmt.OKBLUE}Video Data{pfmt.RESET}")
            print(
                f"[INFO] Frame count: {self.frame_count}\n"
                f"[INFO] FPS: {self.fps}\n"
                f"[INFO] Duration: {self.duration}\n"
                f"[INFO] Frame Width: {self.frame_width}\n"
                f"[INFO] Frame Height: {self.frame_height}"
            )

            self.frame_buf_c = np.array(frame_buf_c, dtype=np.float64)
            self.frame_buf_g = np.array(frame_buf_g, dtype=np.float64)
            self.flow_buf = np.array(flow_buf, dtype=np.float64)

        except Exception as e:
            raise e

    @timeit(
        "Calculating best loop frames",
        ansi=f"{pfmt.BHEADER}",
        end_new_line=True,
    )
    def get_best_loop_color(self):
        best_per_fc = {}
        for f1, f1_diffs in enumerate(self.frame_color_diffs):
            for f2, diff in enumerate(f1_diffs):
                if diff == -1:
                    continue
                fc = f2 - f1 + 1
                if fc in best_per_fc and diff >= best_per_fc[fc][0]:
                    continue

                best_per_fc[fc] = (diff, f1, f2)

        best_overall = (float("infinity"), -1, -1)
        for fc, diff in best_per_fc.items():
            print(f"fc: {fc} | diff: {diff}")
            if diff[0] < best_overall[0]:
                best_overall = diff

        print(f"[INFO] best overall: {best_overall}")

        self.best_start = best_overall[1]
        self.best_end = best_overall[2]

    @timeit(
        "Calculating best loop frames",
        ansi=f"{pfmt.BHEADER}",
        end_new_line=True,
    )
    def get_best_loop_color_flow(self):
        min_diff_per_fc = {}
        for i in range(len(self.frame_color_diffs)):
            for j in range(len(self.frame_color_diffs[i])):
                color_diff = self.frame_color_diffs[i][j]
                flow_cont_diff = self.frame_flow_conts[i][j]
                combined_diff = color_diff * flow_cont_diff
                fc = j - i + 1

                if color_diff == -1 or flow_cont_diff == -1:
                    continue

                if (
                    fc in min_diff_per_fc
                    and combined_diff >= min_diff_per_fc[fc][0]
                ):
                    continue

                min_diff_per_fc[fc] = (combined_diff, i, j)

        best_overall = (float("infinity"), -1, -1)
        for fc, diff in min_diff_per_fc.items():
            print(f"[INFO] fc: {fc} | diff: {diff}")
            if diff[0] < best_overall[0]:
                best_overall = diff

        print(f"[INFO] best overall: {best_overall}")

        self.best_start = best_overall[1]
        self.best_end = best_overall[2]

    @timeit(
        "Writing loop frames",
        ansi=f"{pfmt.BHEADER}",
        end_new_line=True,
    )
    def write_best_loop_frames(self):
        """writes best loop frames to ./tmp_images"""

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
                print(f"[INFO] writing frame ({fc}) to file ({im_path})")
                image_paths.append(im_path)
                cv.imwrite(im_path, frame)

            fc += 1

    @timeit(
        "Creating gif",
        ansi=f"{pfmt.BHEADER}",
        end_new_line=True,
    )
    def create_gif(self):
        """uses imagemagick to create gif using images in ./tmp_images"""

        tmp_dir = "./tmp_images"
        delay = 1 / self.fps
        loop = 0
        gif_path = "./output.gif"
        name, extension = os.path.splitext(gif_path)
        i = 1
        while os.path.exists(gif_path):
            gif_path = name + f"{i}" + extension
            i += 1
        cmd = f"convert -delay {delay} {tmp_dir}/*.png -loop {loop} {gif_path}"
        print(f"[INFO] writing loop to {gif_path}")
        os.system(cmd)

        # cleanup tmp dir
        if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    @timeit(
        "Getting all frames flow similarity",
        ansi=f"{pfmt.BHEADER}",
        end_new_line=True,
    )
    def get_all_frames_flow_conts(self):
        """
        calculates flow continuity between all pairs of frames that would
        create a valid loop

        sets self.frame_flow_conts[i] = [avg_cont(i,0),...,avg_cont(i,n)]
        """
        frame_flow_sims = []
        min_loop_len = math.ceil(self.fps * self.min_duration)
        last_frame_valid = self.frame_count - min_loop_len
        for i in range(last_frame_valid + 1):
            frame_flow_sims.append(self.get_frame_flow_conts(i))

        self.frame_flow_conts = np.array(frame_flow_sims, dtype=np.float64)

    @timeit("Getting flow continuities for loops on frame", include_id=True)
    def get_frame_flow_conts(
        self, i: int
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        calculates average flow continuity for frame i and every
        other valid frame considering the min and max loop duration

        ex: (frames 1 and 4 are frames i and j, respectively)

            frame0 -> frame1 -> frame2 -> frame3 -> frame4 -> frame5
                 flow0  -> flow1 ->  flow2 ->  flow3 ->  flow4


            we want a smooth flow transition from frame3 -> frame4 -> frame1 ->
            frame2

            so we calculate the flow from frame4 to frame 1 and compare
            how similar it is to the flow from frame3 to frame 4 as well
            as from frame1 to frame 2. The assumption is that minimizing
            the difference in flow of these transitions will result in
            a smooth loop

        returns an array flow_conts where flow_conts[j] = avg flow
        continuity of a loop between frames i and j
        """
        flow_conts = np.full(self.frame_count, -1.0, dtype=np.float64)

        min_loop_len = math.ceil(self.fps * self.min_duration)
        max_loop_len = math.ceil(self.fps * self.max_duration)
        first_frame_eligible = i + min_loop_len - 1
        last_frame_eligible = i + max_loop_len - 1
        last_frame_eligible = min(last_frame_eligible, self.frame_count - 1)

        for j in range(first_frame_eligible, last_frame_eligible + 1):
            flow_j_to_i = np.empty(
                (*self.frame_buf_g[0].shape[:2], 2), dtype=np.float32
            )
            cv.calcOpticalFlowFarneback(
                self.frame_buf_g[j],
                self.frame_buf_g[i],
                flow_j_to_i,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            flow_j_to_i = flow_j_to_i.astype(dtype=np.float64)
            loop_to_end_diff = self.get_flow_diff(
                flow_j_to_i, self.flow_buf[j - 1]
            )
            loop_to_start_diff = self.get_flow_diff(
                flow_j_to_i, self.flow_buf[i]
            )
            avg_diff = (loop_to_end_diff + loop_to_start_diff) / 2

            flow_conts[j] = avg_diff

        return flow_conts

    def get_flow_diff(
        self,
        f1: np.ndarray[tuple[Any, Any, Literal[2]], np.dtype[np.float64]],
        f2: np.ndarray[tuple[Any, Any, Literal[2]], np.dtype[np.float64]],
        ang_coef: float = 1.0,
        mag_coef: float = 0.1,
    ) -> np.float64:
        """
        computes average difference between two flow frames

        heuristic is comprised of angular distance and magnitude difference
        of the flow vectors
        ang_coef and mag_coef are coefficients that weight the importance
        of angular distance and magnitude difference in the heuristic
        faster motion of objects in video would likely need a smaller
        mag_ceof value
        a value of 0 would mean flow is the same, a value of 1 means its the
        opposite


        returns flow difference between given frames in range [0,1]
        """
        f1_2d = f1.reshape(-1, 2)
        f2_2d = f2.reshape(-1, 2)

        mag0 = np.sqrt((f1_2d * f1_2d).sum(axis=1))
        mag1 = np.sqrt((f2_2d * f2_2d).sum(axis=1))
        mag_diff = np.abs(mag0 - mag1)

        ang_dist = self.angular_distance(f1_2d, f2_2d)

        diff = ang_dist.mean() * ang_coef + mag_diff.mean() * mag_coef

        if diff > 1:
            return np.float64(1)
        elif diff < 0:
            return np.float64(0)
        else:
            return diff

    def cosine_similarity(
        self,
        x: np.ndarray[tuple[Any, Literal[2]], np.dtype[np.float64]],
        y: np.ndarray[tuple[Any, Literal[2]], np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        computes cosine similarity between two arrays of vectors
        similarity of 1 means perfect match, and -1 means opposite

        returns an array of floats in range [-1,1]
        """
        res = (x * y).sum(axis=1) / (
            np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        )
        # sometimes theres some floating point precision errors
        # makes sure result is within bounds of arccos
        res[res > 1] = 1
        res[res < -1] = -1
        return res

    def angular_distance(
        self,
        v1: np.ndarray[tuple[Any, Literal[2]], np.dtype[np.float64]],
        v2: np.ndarray[tuple[Any, Literal[2]], np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        computes angular distance between two arrays of vectors
        distance of 0 means vectors have the same angle, distance of 1 means
        vectors have the opposite angle

        returns an array of floats in range [0,1]
        """
        return np.arccos(self.cosine_similarity(v1, v2)) / np.pi

    @timeit(
        "Getting all frames color differences",
        ansi=f"{pfmt.BHEADER}",
        end_new_line=True,
    )
    def get_all_frames_color_diffs(self) -> None:
        """
        calculates color difference between all pairs of frames that would
        create a valid loop

        sets self.frame_color_diffs[i] = [avg_diff(i,0),...,avg_diff(i,n)]
        """
        frame_color_diffs = []
        min_loop_len = math.ceil(self.fps * self.min_duration)
        last_frame_eligible = self.frame_count - min_loop_len
        for i in range(last_frame_eligible + 1):
            frame_color_diffs.append(self.get_frame_color_diffs(i))

        self.frame_color_diffs = np.array(frame_color_diffs, dtype=np.float64)

    @timeit("Getting color diffs for frame", include_id=True)
    def get_frame_color_diffs(
        self, i: int
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        calculates average color difference for frame i and every
        other valid frame considering the min and max loop duration

        returns an array frame_diffs where frame_diffs[j] = avg color
        difference between frame[i] and frame[j]
        """
        frame_diffs = np.full(self.frame_count, -1.0, dtype=np.float64)

        min_loop_len = math.ceil(self.fps * self.min_duration)
        max_loop_len = math.ceil(self.fps * self.max_duration)
        first_frame_eligible = i + min_loop_len - 1
        last_frame_eligible = i + max_loop_len - 1
        last_frame_eligible = min(last_frame_eligible, self.frame_count - 1)
        for j in range(first_frame_eligible, last_frame_eligible + 1):
            frame_diffs[j] = self.get_color_diff(i, j)

        return frame_diffs

    def get_frame_color_diff_redmean(
        self, f1_idx: int, f2_idx: int
    ) -> np.float64:
        """
        returns average redmean color difference between given frames
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

    def get_frame_color_diff_CIE_2000(
        self, f1_idx: int, f2_idx: int
    ) -> np.float64:
        """
        returns average CIE 2000 color difference between given frames
        """
        f1, f2 = self.frame_buf_c[f1_idx], self.frame_buf_c[f2_idx]
        f1_lab = cv.cvtColor(f1, cv.COLOR_RGB2Lab)
        f2_lab = cv.cvtColor(f2, cv.COLOR_RGB2Lab)
        delta_e = delta_E_CIE2000(f1_lab, f2_lab)
        return np.mean(delta_e)

    def get_frame_color_diff_gray(
        self, f1_idx: int, f2_idx: int
    ) -> np.float64:
        """
        returns average grayscale color difference between given frames
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
        # vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        vis = img
        cv.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis
