#!/usr/bin/env python3
import numpy as np

from autoencirt.nn import DenseHorseshoeNetwork


def main():
    dnn = DenseHorseshoeNetwork(10, [20, 20, 2])


if __name__ == "__main__":
    main()
