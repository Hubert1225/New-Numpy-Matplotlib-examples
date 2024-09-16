import numpy as np
import numpy.typing as npt

from .structures import LossFunction


class MeansSquareLoss(LossFunction):
    """Means-squared error loss
    (strictly speaking, in our case the sum of squared losses
    is computed)

    Notes
    -----
    `Y_pred` and `Y_true` should always
    have same dimensions.

    """

    def __call__(
        self, Y_pred: npt.NDArray[np.float64], Y_true: npt.NDArray[np.float64 | np.int_]
    ) -> float:
        return np.sum(np.power(Y_pred - Y_true, 2))

    def differentiate(
        self, Y_pred: npt.NDArray[np.float64], Y_true: npt.NDArray[np.float64 | np.int_]
    ) -> float | npt.NDArray[np.float64]:
        return 2 * (Y_pred - Y_true)


class CrossEntropyLoss(LossFunction):
    """Cross entropy loss
    (for multiclass classification problems)

    """

    def __call__(
        self, Y_pred: npt.NDArray[np.float64], Y_true: npt.NDArray[np.float64 | np.int_]
    ) -> float:
        """Calculates cross entropy loss for `C`-classes
        classification problem

        Parameters
        ----------
        Y_pred: array
            for each input, the row vector of length `C` which i-th
            element is the probability of i-th class, i = 0, ..., `C`-1
        Y_true: array
            for each input, the class label - integer
            from range [0, C-1]

        Returns
        -------
        float
            cross entropy loss value

        Notes
        -----
        a little number `eps` is added while extracting logarithm
        in order to avoid log(0)

        """
        eps = 1e-7
        true_classes = Y_true.reshape((-1)).astype("int")
        row_index_array = np.zeros_like(Y_pred) + np.arange(Y_pred.shape[-1])
        true_classes_mask = row_index_array == true_classes.reshape(-1, 1)
        return -1 * np.sum(np.log(Y_pred[true_classes_mask] + eps))

    def differentiate(
        self, Y_pred: npt.NDArray[np.float64], Y_true: npt.NDArray[np.float64 | np.int_]
    ) -> float | npt.NDArray[np.float64]:
        """Calculates cross entropy loss gradient
        for `C`-classes classification problem

        Parameters
        ----------
        Y_pred: array
            for each input, the row vector of length `C` which i-th
            element is the probability of i-th class, i = 0, ..., `C`-1
        Y_true: array
            for each input, the class label - integer
            from range [0, C-1]

        Returns
        -------
        float
            cross entropy loss gradient

        Notes
        -----
        a little number `eps` is added to denominator to avoid
        zero division

        """
        eps = 1e-7
        true_classes = Y_true.reshape((-1)).astype("int")
        row_index_array = np.zeros_like(Y_pred) + np.arange(Y_pred.shape[-1])
        true_classes_mask = (row_index_array == true_classes.reshape(-1, 1)).astype(
            "int"
        )
        return -1 * true_classes_mask / (Y_pred + eps)
