import numpy as np
from scipy.stats import mode


class EpsilonCubeNeighborsClassifier:
    """
    Epsilon Cube Neighbors Classifier

    Parameters
    ----------
    sides : list
        The sides of the cube in each dimension
    outlier_label : int
        The label for the outlier class

    Attributes
    ----------
    sides : numpy.ndarray
        The sides of the cube in each dimension
    n_dims : int
        The number of dimensions
    outlier_label : int
        The label for the outlier class
    X : numpy.ndarray
        The training data
    y : numpy.ndarray
        The training labels
    """

    def __init__(self, sides, outlier_label) -> None:
        self.sides = np.abs(np.array(sides).reshape(1, -1))  # (1, n_dims)
        self.n_dims = len(sides)
        self.outlier_label = int(outlier_label)

    def fit(self, X, y):
        assert len(X) == len(y)
        assert X.shape[1] == self.n_dims
        self.X = X  # (n_train, n_dims)
        self.y = y  # (n_train, )

    def is_inbound(self, point, lower, upper):
        """
        Check if the point is in the range [lower, upper]
        """
        if lower <= point <= upper:
            return True
        return False

    def check_outlier(self, inds):
        """
        Check if the neighbors are outliers
        """
        return mode(self.y[inds])[0] if inds.size > 0 else self.outlier_label

    def predict(self, X):  # (n_test, n_dims)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)  # (1, n_dims)
        assert X.shape[1] == self.n_dims
        boundaries_min = X - np.repeat(self.sides, len(X), axis=0)  # (n_test, n_dims)
        boundaries_max = X + np.repeat(self.sides, len(X), axis=0)  # (n_test, n_dims)
        v_inbound = np.vectorize(self.is_inbound)
        neighbor_inds = []
        for i in range(len(X)):
            boundary_min = np.repeat(
                boundaries_min[i].reshape(1, -1), len(self.y), axis=0
            )
            boundary_max = np.repeat(
                boundaries_max[i].reshape(1, -1), len(self.y), axis=0
            )
            assert self.X.shape == boundary_min.shape
            neighbor_inds.append(
                np.where(np.all(v_inbound(self.X, boundary_min, boundary_max), axis=1))[
                    0
                ].flatten()
            )
        return [self.check_outlier(neighbor_inds[i]) for i in range(len(X))]
