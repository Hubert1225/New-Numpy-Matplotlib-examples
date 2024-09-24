import numpy as np
import numpy.typing as npt


def rgb2gray(rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Converts an RGB image to grayscale

    Function adapted from
    https://www.askpython.com/python-modules/matplotlib/convert-rgb-to-grayscale

    Parameters
    ----------
    rgb : 3D array
        RGB image channels

    Returns
    -------
    2D array
        image in grayscale

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144]).astype(np.uint8)
