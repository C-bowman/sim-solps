
from numpy import ones, nan, full


class Thomson(object):
    """
    Synthetic diagnostic model for Thomson scattering.

    :param R: \
        Radius positions of the sampling points as a 2D numpy array of shape
        ``(num_channels, num_samples)``.

    :param z: \
        z-positions of the sampling points as a 2D numpy array of shape
        ``(num_channels, num_samples)``.

    :param weights: \
        weighting values of the sampling points as a 2D numpy array of shape
        ``(num_channels, num_samples)``.

    :param interface: \
        An instance of the SolpsInterface class.
    """
    def __init__(self, R, z, weights, interface):
        self.n_channels, self.n_samples = R.shape
        self.R = R
        self.z = z
        self.weights = weights / weights.sum(axis=1)[:, None]
        self.interface = interface

        # first find out which sample points are actually inside the mesh
        in_mesh = self.interface.mesh.interpolate(
            self.R,
            self.z,
            ones(self.interface.mesh.n_vertices)
        )
        # use this to find what total probability of the instrument function
        # is inside the mesh for each channel
        probs = (in_mesh*self.weights).sum(axis=1)
        # only make predictions for channels where over 90%
        # of the probability is inside the mesh
        self.predicted_channels = (probs > 0.9).nonzero()
        # now re-normalise the weights to account of points outside the mesh
        self.weights = (in_mesh * self.weights)[self.predicted_channels, :]
        self.weights = self.weights / self.weights.sum(axis=1)[:, None]
        # discard sample point data for channels outside mesh
        self.R = self.R[self.predicted_channels]
        self.z = self.z[self.predicted_channels]

    def predict(self):
        te_samples = self.interface.get('te', self.R, self.z)
        ne_samples = self.interface.get('ne', self.R, self.z)

        ne_predictions = (ne_samples * self.weights).sum(axis=1)
        te_predictions = (te_samples * ne_samples * self.weights).sum(axis=1) / ne_predictions

        ne = full(self.n_channels, fill_value=nan)
        ne[self.predicted_channels] = ne_predictions
        te = full(self.n_channels, fill_value=nan)
        te[self.predicted_channels] = te_predictions

        return ne, te

    @classmethod
    def mastu_core(cls):
        pass


