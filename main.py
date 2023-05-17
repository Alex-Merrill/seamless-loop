import argparse
import os

import looper


def parse_args():
    """
    Parses input arguments with basic check

    returns arguments (min_duration, max_duration, path)
    """

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-mind",
        "--min-duration",
        help="minimum duration of final loop",
        type=int,
    )
    arg_parser.add_argument(
        "-maxd",
        "--max-duration",
        help="maximum duration of final loop",
        type=int,
    )
    arg_parser.add_argument("source", help="path to video source")

    args = arg_parser.parse_args()
    src = args.source
    min_d = args.min_duration if args.min_duration else 3
    max_d = args.max_duration if args.max_duration else 5

    if not os.path.isfile(src):
        raise FileNotFoundError("file not found, check source")

    return min_d, max_d, src


def main():
    """
    Gets input args and passes them to looper, starts algorithm.
    """

    min_d, max_d, src = parse_args()
    loop = looper.Looper(min_d, max_d, src)
    loop.Start()


if __name__ == "__main__":
    main()
