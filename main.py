import argparse
import nn


class FCNet(nn.Network):
    def __init__(self):
        pass


def train():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int, default=20)
    args = parser.parse_args()

