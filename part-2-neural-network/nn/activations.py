import numpy as np
import numpy.typing as npt

from .structures import Layer


class SigmoidActivation(Layer):
    """
    Attributes
    ----------
    last_activation_: float | array | None
        output from the last `forward` call
    """

    name = "sigmoid"

    def _initialize_params(self, *args, **kwargs) -> dict[str, npt.NDArray[np.float_]]:
        return dict()

    def __init__(self):
        super().__init__()
        self.last_activation_: float | npt.NDArray[np.float] | None = None

    def forward(self, X: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        self.last_activation_ = np.exp(X) / (np.exp(X) + 1)
        return self.last_activation_

    def backward(self, dY: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return dY * self.last_activation_ * (1 - self.last_activation_)


class ReLUActivation(Layer):
    """
    Attributes
    ----------
    last_mask_: array | None
        binary array with values:
        1 if corresponding element in the last `forward` call input
        was greater than 0,
        0 otherwise
    """

    name = "ReLU"

    def _initialize_params(self, *args, **kwargs) -> dict[str, npt.NDArray[np.float_]]:
        return dict()

    def __init__(self):
        super().__init__()
        self.last_mask_: npt.NDArray[np.float] | None = None

    def forward(self, X: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        self.last_mask_ = (X > 0).astype("int8")
        return X * self.last_mask_

    def backward(self, dY: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        return self.last_mask_.astype("float") * dY


class SoftmaxActivation(Layer):
    """
    Attributes
    ----------
    last_activation_: float | array | None
        output from the last `forward` call
    """

    name = "softmax"

    def _initialize_params(self, *args, **kwargs) -> dict[str, npt.NDArray[np.float_]]:
        return dict()

    def __init__(self):
        super().__init__()
        self.last_activation_: float | npt.NDArray[np.float] | None = None

    def forward(self, X: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        X_exps = np.exp(X)
        self.last_activation_ = X_exps / np.sum(X_exps, axis=-1, keepdims=True)
        return self.last_activation_

    def backward(self, dY: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        # stack of derivatives matrices
        # matrix element [i,j] - d(activation_i)/d(input_j)
        # = -y_i*y_j if i != j
        # else y_i*(1-y_i)
        n_inputs = self.last_activation_.shape[-1]
        activations = self.last_activation_.reshape((-1, n_inputs))
        activations_rowlike = self.last_activation_.reshape((-1, 1, n_inputs))
        activations_columnlike = self.last_activation_.reshape((-1, n_inputs, 1))
        derivatives = -1 * np.matmul(activations_columnlike, activations_rowlike)
        derivatives[:, np.arange(n_inputs), np.arange(n_inputs)] = activations * (
            1 - activations
        )
        # apply chain rule
        derivatives *= dY.reshape(-1, n_inputs, 1)
        # return aggregated gradient for each x_i
        return np.sum(derivatives, axis=-2).reshape(self.last_activation_.shape)
