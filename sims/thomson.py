
from numpy import array, ones, nan, full, ndarray


class ThomsonScattering(object):
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
        An instance of the ``SolpsInterface`` class. If the interface cannot be
        passed when creating the instance of ``ThomsonScattering``, it can be
        specified afterward using the ``update_interface`` method.

    :param measurements: \
        A dictionary containing Thomson-scattering measurement data to which
        the synthetic instrument predictions will be compared. The data should
        be given as 1D numpy arrays under the keys 'te_data', 'te_err', 'ne_data'
        and 'ne_err'. This keyword argument is required in order to use the
        ``log_likelihood`` method.
    """
    def __init__(self, R, z, weights, interface=None, measurements=None):
        self.shape = R.shape
        self.n_channels, self.n_samples = self.shape
        self.R = R
        self.z = z
        self.weights = weights / weights.sum(axis=1)[:, None]

        # attributes which depend on the interface
        self.interface = None
        self.predicted_channels = None
        self.p = None
        self.R_p = None
        self.z_p = None
        if interface is not None:
            self.update_interface(interface)

        # attributes for storing experimental data
        if measurements is not None:
            self.te_data = measurements['te_data']
            self.te_err = measurements['te_err']
            self.ne_data = measurements['ne_data']
            self.ne_err = measurements['ne_err']
            self.check_measurements()

    def check_measurements(self, measurements):
        data_dict = {}
        for key in ['te_data', 'te_err', 'ne_data', 'ne_err']:
            typ = type(measurements[key])
            if any([typ is t for t in [list, tuple]]):
                data_dict[key] = array(measurements[key]).flatten()
            elif typ is ndarray:
                data_dict[key] = measurements[key].flatten()
            else:
                raise TypeError(
                    f"""
                    The objected stored under the '{key}' key of the 'measurements'
                    keyword argument should be of type 'ndarray', but instead type
                    {typ} was found.
                    """
                )

        # now check all the arrays are the same length
        sizes = [a.size for a in data_dict.values()]
        if any([sizes[0] != s for s in sizes]):
            raise ValueError(
                f"""
                The data stored under the following keys:
                ...
                """
            )

    def update_interface(self, interface):
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
        # only make predictions for channels where over 95%
        # of the probability is inside the mesh
        self.predicted_channels = (probs > 0.95).nonzero()
        # now re-normalise the weights to account of points outside the mesh
        self.p = (in_mesh * self.weights)[self.predicted_channels, :]
        self.p = self.p / self.p.sum(axis=1)[:, None]
        # discard sample point data for channels outside mesh
        self.R_p = self.R[self.predicted_channels, :]
        self.z_p = self.z[self.predicted_channels, :]

    def predict(self):
        te_samples = self.interface.get('te', self.R, self.z)
        ne_samples = self.interface.get('ne', self.R, self.z)

        ne_predictions = (ne_samples * self.p).sum(axis=1)
        te_predictions = (te_samples * ne_samples * self.p).sum(axis=1) / ne_predictions

        ne = full(self.n_channels, fill_value=nan)
        ne[self.predicted_channels] = ne_predictions
        te = full(self.n_channels, fill_value=nan)
        te[self.predicted_channels] = te_predictions
        return ne, te

    def log_likelihood(self):
        pass

    @classmethod
    def mastu_core(cls):
        pass


