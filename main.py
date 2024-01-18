import argparse
import os

import looper


def parse_args():
    """
    Parses input arguments with basic check

    returns arguments (min_duration, max_duration, path)
    """

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg_parser.add_argument(
        "-mind",
        "--min-duration",
        help="minimum duration of final loop",
        nargs="?",
        default=3,
        const=3,
        type=int,
    )
    arg_parser.add_argument(
        "-maxd",
        "--max-duration",
        help="maximum duration of final loop",
        nargs="?",
        default=5,
        const=5,
        type=int,
    )
    colors = arg_parser.add_mutually_exclusive_group()
    colors.add_argument(
        "-g",
        "--grayscale",
        help="process video in grayscale",
        action="store_true",
    )
    colors.add_argument(
        "-cd",
        "--colordifference",
        choices=["redmean", "CIE 2000"],
        help="choose color difference technique",
        nargs="?",
        default="redmean",
        const="redmean",
        type=str,
    )
    arg_parser.add_argument("source", help="path to video source")

    args = arg_parser.parse_args()
    src = args.source
    min_d = args.min_duration
    max_d = args.max_duration
    gray = args.grayscale
    color_diff = args.colordifference

    if not os.path.isfile(src):
        raise FileNotFoundError("file not found, check source")

    return min_d, max_d, gray, color_diff, src


def main():
    """
    Gets input args and passes them to looper, starts algorithm.
    """

    min_d, max_d, gray, color_diff, src = parse_args()
    loop = looper.Looper(min_d, max_d, gray, color_diff, src)
    loop.Start()


if __name__ == "__main__":
    main()
