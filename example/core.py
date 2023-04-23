import sys
from os.path import pardir, join, abspath, dirname


def appendParendPath():
    root_path = abspath(join(
        dirname(__file__), pardir))
    sys.path.append(root_path)


if __name__ == "__main__":
    appendParendPath()
