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


def rotate_image(image: npt.NDArray[np.uint8], phi: float) -> npt.NDArray[np.uint8]:
    """Produces rotated version of a given image.

    Parameters
    ----------
    image : 2D array or 3D array
        image to be rotated
        (if 3D, of shape: (height, width, channels) )
    phi : float
        rotation angle, in radians
        (counter-clockwise rotation)

    Returns
    -------
    2D array or 3D array:
        rotated image

    """

    # rotation matrix
    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

    # obtain grid of pixels' coordinates
    n_rows, n_cols = image.shape[:3]
    x, y = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    coord = np.stack((x, y)).astype(np.int32)
    coord = np.expand_dims(np.transpose(coord, (1, 2, 0)), axis=-1)

    # obtain coordinates of rotated image
    coord_rot = np.matmul(R, coord)
    x_rot = coord_rot[:, :, 0]
    y_rot = coord_rot[:, :, 1]

    # calculate how the image's center is displaced in the rotated image
    # then align rotated coordinates
    # (we want the center of the image to be center of the rotated image)
    center = np.array([[n_rows // 2], [n_cols // 2]], dtype=np.int32)
    center_rot = np.matmul(R, center).astype(np.int32)
    center_displacement = center_rot - center
    x_rot = x_rot - center_displacement[0, 0]
    y_rot = y_rot - center_displacement[1, 0]

    # we don't want the "wrapping effect" in the result
    # set temporarily to -1
    x_rot[np.logical_or(x_rot < 1 - n_rows, x_rot > n_rows - 1)] = -1
    y_rot[np.logical_or(y_rot < 1 - n_cols, y_rot > n_cols - 1)] = -1

    # create the result image
    output = image[x_rot, y_rot]
    x_invalid_coords = np.logical_or(x_rot < 0, x_rot > n_rows - 1)
    y_invalid_coords = np.logical_or(y_rot < 0, y_rot > n_cols - 1)
    invalid_coords = np.logical_or(x_invalid_coords, y_invalid_coords)
    output[invalid_coords] = 0

    return output
