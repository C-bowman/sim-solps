
from numpy import array, ones, nan, full, ndarray
from sims.likelihoods import gaussian_likelihood


class ThomsonScattering(object):
    """
    Synthetic instrument model for Thomson scattering.

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
        specified later using the ``update_interface`` method.

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

        # now check all the arrays have the correct length
        for key, v in data_dict.items():
            if v.size != self.n_channels:
                raise ValueError(
                    f"""
                    The instrument was specified to have {self.n_channels} channels, but
                    the given {key} array has size {v.size}.
                    """
                )

    def update_interface(self, interface):
        """
        Set the SOLPS data which will be used by the instrument model by passing
        an instance of ``SolpsInterface`` from the ``sims.interface`` module.

        :param interface: \
            An instance of ``SolpsInterface`` from the ``sims.interface`` module.
        """
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
        self.p = (in_mesh * self.weights)[self.predicted_channels, :].squeeze()
        self.p = self.p / self.p.sum(axis=1)[:, None]
        # discard sample point data for channels outside mesh
        self.R_p = self.R[self.predicted_channels, :].squeeze()
        self.z_p = self.z[self.predicted_channels, :].squeeze()

    def predict(self):
        """
        Calculate predictions of the electron temperature and density measurements
        made by the instrument for the given SOLPS results.

        :return: \
            Predictions of the electron density and temperature as two numpy arrays.
        """
        te_samples = self.interface.get('electron temperature', self.R_p, self.z_p)
        ne_samples = self.interface.get('electron density', self.R_p, self.z_p)

        ne_predictions = (ne_samples * self.p).sum(axis=1)
        te_predictions = (te_samples * ne_samples * self.p).sum(axis=1) / ne_predictions

        ne = full(self.n_channels, fill_value=nan)
        ne[self.predicted_channels] = ne_predictions
        te = full(self.n_channels, fill_value=nan)
        te[self.predicted_channels] = te_predictions
        return ne, te

    def log_likelihood(self, likelihood=gaussian_likelihood):
        """
        Calculate the log-likelihood of the experimental data for the given
        SOLPS data.

        :param likelihood: \
            The likelihood function used in the calculation. By default, a
            Gaussian likelihood is used. Other likelihoods can be imported
            from the ``sims.likelihoods`` module and passed to change which
            likelihood is used.

        :return: The log-likelihood
        """
        ne_prediction, te_prediction = self.predict()
        te_ll = likelihood(
            self.te_data[self.predicted_channels],
            self.te_err[self.predicted_channels],
            te_prediction[self.predicted_channels]
        )
        ne_ll = likelihood(
            self.ne_data[self.predicted_channels],
            self.ne_err[self.predicted_channels],
            ne_prediction[self.predicted_channels]
        )
        return te_ll + ne_ll


