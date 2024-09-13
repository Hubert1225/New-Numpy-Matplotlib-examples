"""This module provides classes serving base structure
for building neural networks

"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class Layer(ABC):
    """Base class for a layer of neural network.

    Each layer represents transformation of data
    X -> Y
    using a set of parameters values that can be trained
    using gradient descent.

    Attributes
    ----------
    name : str
        The name of the layer type
    params_ : dict[str, npt.NDArray[np.float_]]
        The dict storing values of layer parameters,
        keys are strings with names of parameters
        and values are NumPy arrays containing values
        of parameters
    grads_ : dict[str, npt.NDArray[np.float_]]
        The dict storing gradient values for each parameter,
        the set of keys is always the same as set of keys
        of params_ dict,
        and for each key the value is a NumPy array of
        exactly the same shape as the shape of corresponding
        parameter which contains gradient values for this
        parameter

    """

    name: str

    @abstractmethod
    def _initialize_params(self, *args, **kwargs) -> dict[str, npt.NDArray[np.float_]]:
        """Creates values to initialize self.params_ dict

        Returns
        -------
        dict[str, npt.NDArray[np.float_]]
            The dict which keys should be strings (names of parameters)
            and for each key the value should be NumPy array with initial
            values of the parameter
        """
        pass

    def __init__(self, *args, **kwargs):
        self.params_: dict[str, npt.NDArray[np.float_]] = self._initialize_params(
            *args, **kwargs
        )
        self.grads_: dict[str, npt.NDArray[np.float_]] = {
            param_name: np.zeros_like(param_arr)
            for param_name, param_arr in self.params_.items()
        }

    @abstractmethod
    def forward(self, X: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Performs forward propagation of the layer
        on the input data
        X -> Y

        Parameters
        ----------
        X : array
            input data

        Returns
        -------
        Y : array
            layer's output for the given input data

        """
        pass

    @abstractmethod
    def backward(self, dY: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Given the gradient of layer's output, performs
        gradient backpropagation.

        dY -> dX, dP
        (X - layer's input, P - layer's parameters)

        This method must do two things:
         - compute gradients of layer's parameters dP and save it
         in the `self.grads_` variable
         - compute gradients of layer's input dX and return it

        Parameters
        ----------
        dY : array
            An array of the same shape as the layer's output,
            containing gradient values of the output

        Returns
        -------
        dX : array
            An array of the same shape as the layer's input,
            containing gradient values of the input

        """
        pass

    def update(self, lr: float) -> None:
        """Updates the layer's parameters' values with gradient descent
        using gradient values saved in the `self.grads_` variable

        Parameters
        ----------
        lr : float
            The learning rate - the coefficient which is multiplied
            by gradient values to obtain the update step

        """
        for param_name, param_grad in self.grads_.items():
            self.params_[param_name] = self.params_[param_name] - lr * param_grad


class LayersSequence:
    """Sequence of layers
    (the output of i-th layer is the input of (i+1)th layer)

    Attributes
    ----------
    layers : list[Layer]
        list of instances of `Layer` subclasses

    Notes
    -----
    The user of this class should pick layers and dimensions of their
    inputs so that the shape of output of i-th layer is the same as
    shape of input of (i+1)th layer

    """

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def forward(self, X: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Performs forward propagation on layers sequentially.

        The output of layer at index i in `self.layers` is the input
        of layer at index (i+1) in `self.layers`. Therefore, the shape of
        input X should match the input shape of the layer at index 0
        and the output will have the shape as the output of the last layer.

        Parameters
        ----------
        X : array
            input data

        Returns
        -------
        Y : array
            output from the sequence of layers

        """
        Y = X
        for layer in self.layers:
            Y = layer.forward(Y)
        return Y

    def backward(self, dY: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """Performs backward propagation on the sequence
        of layers

        The gradient of sequence output is passed to the last layer,
        then propagated to the second to last, and so on. Therefore,
        the shape of gradient passed should have the same shape
        as the last layer output and the resulted gradient after
        propagation will have shape of input of the layer at the index 0
        in `self.layers`.

        Parameters
        ----------
        dY : gradient values of the sequence output

        Returns
        -------
        dX : gradient values of the sequence input

        """
        dX = dY
        for layer in reversed(self.layers):
            dX = layer.backward(dX)
        return dX

    def update(self, lr: float) -> None:
        """Updates each layer's parameters with gradient descent
        using gradient values obtained with backpropagation.

        Parameters
        ----------
        lr : float
            The learning rate - the coefficient which is multiplied
            by gradient values to obtain the update step

        """
        for layer in self.layers:
            layer.update(lr=lr)


class LossFunction(ABC):
    """Base class for classes representing loss functions
    whose values are minimized while training networks

    """

    @abstractmethod
    def __call__(
        self, Y_pred: npt.NDArray[np.float_], Y_true: npt.NDArray[np.float_ | np.int_]
    ) -> float:
        """Returns loss value

        Parameters
        ----------
        Y_pred: array
            predicted values
        Y_true:
            ground-truth values

        Returns
        -------
        float
            loss value

        Notes
        -----
        Y_pred and Y_true does not to always have the same dimensions
        (e.g. Y_pred may be classes probabilities and Y_true just
        ground-truth class labels). The choice of how Y_pred and Y_true
        have to look like is given to child classes developer

        """
        pass

    @abstractmethod
    def differentiate(
        self, Y_pred: npt.NDArray[np.float_], Y_true: npt.NDArray[np.float_ | np.int_]
    ) -> float | npt.NDArray[np.float_]:
        """Calculates the loss function gradient over predicted
        values

        Parameters
        ----------
        Y_pred: array
            predicted values
        Y_true: array
            ground truth values

        Returns
        -------
        array
            the array of the same shape as Y_pred,
            containing the gradient values for Y_pred

        """
        pass
