import os
import argparse

import looper


def parse_args():
    """
    Parses input arguments with basic check

    returns arguments (min_duration, max_duration, path)
    """

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-mind', '--min_duration',
                            help='minimum duration of final loop',
                            type=int)
    arg_parser.add_argument('-maxd', '--max_duration',
                            help='maximum duration of final loop',
                            type=int)
    arg_parser.add_argument('path', help='path to video file')

    arg_parser = parse_args()
    path = arg_parser.path
    min_d = arg_parser.min_duration if arg_parser.min_duration else 3
    max_d = arg_parser.max_duration if arg_parser.max_duration else 5

    if not os.path.isfile(path):
        raise FileNotFoundError("file not found, check path")

    return min_d, max_d, path


def main():
    """
    Gets input args and passes them to looper, starts algorithm.
    """

    min_d, max_d, path = parse_args()
    loop = looper.Looper(min_d, max_d, path)
    loop.Start()


if __name__ == "__main__":
    main()
