import numpy as np
import numpy.typing as npt

from .structures import Layer


class LinearLayer(Layer):
    """Linear layer of a neural network.

    Performs linear transformation of data:
    X -> Y
    which is the matrix multiplication:
    Y = XW + b
    W - matrix of weights
    b - vector of biases

    Notes
    -----
    This layer stores values of last input X
    in the `last_X_` field (i.e. X values from last
    `forward` method call) as they are necessary to
    calculate gradients. Therefore, when `backward`
    method is called, it refers to X values from
    `last_X_` field (if the user does not change it
    manually, the field stores the X values from last
    `forward` call)

    """

    name = "Linear"

    def __init__(self, n_inputs: int, n_outputs: int):
        self.n_inputs: int = n_inputs
        self.n_outputs: int = n_outputs
        self.last_X_: npt.NDArray | None = None
        super().__init__()

    def _initialize_params(self, *args, **kwargs) -> dict[str, npt.NDArray[np.float64]]:
        weights = np.random.uniform(
            low=-1.0, high=1.0, size=(self.n_inputs, self.n_outputs)
        )
        biases = np.random.uniform(low=-3.0, high=3.0, size=(self.n_outputs,))
        return {"weights": weights, "biases": biases}

    def forward(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.last_X_ = X
        return np.matmul(X, self.params_["weights"]) + self.params_["biases"]

    def backward(self, dY: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # dw_ij = x_i * d_j
        # if there's more dimensions, they are averaged
        d_weights = np.matmul(
            self.last_X_.reshape((-1, self.n_inputs, 1)),
            dY.reshape((-1, 1, self.n_outputs)),
        ).reshape((-1, self.n_inputs, self.n_outputs))
        d_weights = np.mean(d_weights, axis=0)
        # db_i = dY_i
        # if there's more dimensions, they are summed
        d_biases = np.mean(dY.reshape(-1, self.n_outputs), axis=0)
        # dx_i = sum_j(dy_j * w_ij)
        dX = np.matmul(
            self.params_["weights"], dY.reshape((-1, self.n_outputs, 1))
        ).reshape(self.last_X_.shape)
        self.grads_ = {"weights": d_weights, "biases": d_biases}
        return dX
