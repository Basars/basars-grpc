import tensorflow as tf
import numpy as np

from functools import reduce


def grpc_preprocess(buffer, dtype=np.uint8, shape=None):
    pixels = np.frombuffer(buffer, dtype=dtype)
    if shape is not None:
        size = reduce(lambda a, b: a * b, shape)
        if pixels.shape[-1] != size:
            return None

        pixels = np.reshape(pixels, shape)
    pixels = pixels[tf.newaxis, ...]
    return pixels


def grpc_postprocess(pixels):
    pixels = np.reshape(pixels, (-1,))
    pixels = bytes(pixels)
    return pixels
