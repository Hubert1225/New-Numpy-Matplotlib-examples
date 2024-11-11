import numpy as np
import numpy.typing as npt


def rgb2gray(rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Converts an RGB image to grayscale

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
    n_rows, n_cols = image.shape[:2]
    x, y = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    coord = np.stack((x, y)).astype(np.int32)
    coord = np.expand_dims(np.transpose(coord, (1, 2, 0)), axis=-1)

    # obtain coordinates of rotated image
    coord_rot = np.matmul(R, coord).astype(np.int32)
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

    return np.squeeze(output)


def conv2d(
    image: npt.NDArray[np.uint8],
    kernel: npt.NDArray[np.float64],
    convert_to_uint8: bool = True,
) -> npt.NDArray[np.uint8]:
    """Performs 2D convolution on given image
    and given kernel

    If the image has shape (n,m) and the kernel has shape (n_k, m_k),
    then the output shape will be (n - n_k + 1, m - m_k + 1).

    Parameters
    ----------
    image: 2D array
        image to be convolved
    kernel: 2D array
        convolution kernel
    convert_to_uint8: bool
        whether to return the result array of type np.uint8
        (array of type np.float64 is returned otherwise)

    Returns
    -------
    2D array: convolved image

    Notes
    -----
    This is not the most efficient way to compute this;
    for more efficient ways, see:
    https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy

    """

    # convert image to np.float64
    image = image.astype(np.float64)

    # create 4D array of shape (n_k, m_k, n, m),
    # where (n, m) - shape of the image,
    #       (n_k, m_k) - shape of the kernel,
    # which can be thought as a matrix of the same shape as kernel,
    # which contains a copy of the image in each cell
    image_reshaped = image.reshape((1, 1, image.shape[0], image.shape[1]))
    image_repeated = np.repeat(
        np.repeat(image_reshaped, kernel.shape[0], 0), kernel.shape[1], 1
    )

    # multiply all values in each cell by the value
    # of the corresponding kernel's cell
    image_repeated_weighted = image_repeated * kernel.reshape(
        kernel.shape[0], kernel.shape[1], 1, 1
    )

    # allocate output image
    output_image = np.zeros(np.subtract(image.shape, kernel.shape) + 1)

    # compute convolution

    # the image's copy from the cell [i, j] contribute
    # but without rows at indexes: 0, 1, ..., i-1 and n - n_k + 1 + i, ... n - 1
    # and without columns at indexes: 0, 1, ..., j-1 and m - m_k + 1 + j, ..., m - 1
    n_rows, n_cols = image.shape
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            output_image = (
                output_image
                + image_repeated_weighted[
                    i,
                    j,
                    i : (n_rows - kernel.shape[0] + i + 1),
                    j : (n_cols - kernel.shape[1] + j + 1),
                ]
            )

    # return image converted back to np.uint8
    if convert_to_uint8:
        output_image = output_image.astype(np.uint8)
    return output_image


def gaussian2d_pdf(
    x: np.ndarray,
    mu: np.ndarray = np.array([0, 0]),
    sigma: np.ndarray = np.array([[1, 0], [0, 1]]),
) -> float:
    return float(
        (1 / 2 * np.pi)
        * (1 / np.sqrt(np.linalg.det(sigma)))
        * np.exp(
            (-1 / 2)
            * np.matmul(
                np.matmul(x - mu, np.linalg.inv(sigma)), (x - mu).reshape(-1, 1)
            )
        )
    )


def detect_edges(image: np.ndarray, denoising_filter: np.ndarray) -> np.ndarray:
    """Given an image, produces the edge image (pixels intensity
    corresponds to the possibility of an edge presence)

    If given a 3-channel image, the image is converted to grayscale.
    Before the edge detection, the image is denoised using the given denoising
    filter.
    Edges are detected by summing up absolute values of horizontal
    and vertical gradients at each pixel.

    Parameters
    ----------
    image: 2D or 3D array
        image to detect edges in
    denoising_filter: 2D array
        denoising convolution kernel

    Returns
    -------
    2D array: edge image

    """

    # step 1: if RGB image, convert to grayscale
    if image.ndim == 3:
        image = rgb2gray(image)

    # step 2: filtering
    image_filtered = conv2d(image, denoising_filter)

    # step 3: vertical edge detection
    vertical_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    vertical_edges = np.abs(
        conv2d(image_filtered, vertical_kernel, convert_to_uint8=False)
    )

    # step 4: horizontal edge detection
    horizontal_kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    horizontal_edges = np.abs(
        conv2d(image_filtered, horizontal_kernel, convert_to_uint8=False)
    )

    # step 5: sum up results from vertical direction and horizontal direction
    # and normalize values to the interval [0; 255]
    edges = vertical_edges + horizontal_edges
    edges = (edges - np.amax(edges)) / (np.amax(edges) - np.amin(edges)) * 255

    return edges.astype(np.uint8)
