import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .structures import LayersSequence


def visualize_weights(
    network: LayersSequence,
    inds_to_visualize: list[int],
    params_to_visualize: list[str],
    title: str = "Network weights",
    ncols: int = 3,
) -> None:
    """Plots values of parameters
    of given network

    Parameters
    ----------
    network: LayersSequence
        network which layers' parameters are to be
        visualized
    inds_to_visualize: list[int]
        list of indexes, for each index the parameter of
        the layer at this index in `network.layers` is visualized
    params_to_visualize: list[str]
        list of the same length as `inds_to_visualize`,
        for each index its the key in the layer's `params_` dict
        that is to be visualized
    title: str
        the main title of the whole figure,
        defualt `"Network weights"`
    ncols: int
        number of columns in the axes grid,
        default 3

    """
    nlayers = len(inds_to_visualize)
    nrows = np.ceil(nlayers / ncols).astype("int")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    fig.suptitle(title)
    for ind, ax, param_name in zip(
        inds_to_visualize, axes.ravel(), params_to_visualize
    ):
        layer = network.layers[ind]
        ax.set_title(f"Values of {param_name}\nlayer {ind}: {layer.name}")
        params_vals = layer.params_[param_name]
        params_vals = (
            params_vals.reshape(1, -1) if params_vals.ndim == 1 else params_vals
        )
        plot = ax.imshow(params_vals)
        fig.colorbar(plot, ax=ax, location="right")


def visualize_activations(
    network: LayersSequence,
    inds_to_visualize: list[int],
    X: npt.NDArray[np.float_],
    ncols: int = 3,
    title: str = "Network activations",
) -> None:
    """Plots values of activations
    of given network for given data

    Parameters
    ----------
    network: LayersSequence
        network which layers' activations are to be
        visualized
    inds_to_visualize: list[int]
        list of indexes, for each index the acttivations of
        the layer at this index in `network.layers` is visualized
    X: array
        data that is passed to the `network` to get
        activations that are to be visualized
    ncols: int
        number of columns in the axes grid,
        default 3
    title: str
        the main title of the whole figure,
        default `"Network activations"`

    """
    nlayers = len(inds_to_visualize)
    nrows = np.ceil(nlayers / ncols).astype("int")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    fig.suptitle(title)
    for ind, ax in zip(inds_to_visualize, axes.ravel()):
        layer = network.layers[ind]
        temp_network = LayersSequence(layers=network.layers[: (ind + 1)])
        ax.set_title(f"Activations of layer {ind}: {layer.name}")
        mean_activations = np.mean(temp_network.forward(X), axis=0)
        plot = ax.imshow(mean_activations.reshape(1, -1), cmap="bwr")
        fig.colorbar(plot, ax=ax, location="right")
