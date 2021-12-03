
from numpy import array, concatenate, zeros, full
from scipy.io import netcdf
from tokamesh import TriangularMesh
from tokamesh.construction import remove_duplicate_vertices
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class SolpsInterface(object):
    """
    A class which provides an interface to the results of SOLPS-ITER
    simulations stored in a balance.nc file.

    :param balance_filepath: \
        A path to a balance.nc file.
    """
    def __init__(self, balance_filepath):
        with netcdf.netcdf_file(balance_filepath, 'r') as solps:
            # TODO - check we're cutting out correct cells from den_n
            den_i = solps.variables['na'].data.copy()
            trim = (slice(None), slice(None), slice(0, den_i.shape[-1]))
            den_n = solps.variables['dab2'].data[trim].copy()
            tab2 = solps.variables['tab2'].data[trim].copy() / 1.602e-19

            self.ne = solps.variables['ne'].data.copy().flatten()
            self.te = solps.variables['te'].data.copy().flatten() / 1.602e-19
            self.ti = solps.variables['ti'].data.copy().flatten() / 1.602e-19
            self.vol = solps.variables['vol'].data.copy().flatten()
            self.dmb2 = solps.variables['dmb2'].data[trim].copy().flatten()
            self.tmb2 = solps.variables['tmb2'].data[trim].copy().flatten() / 1.602e-19

            # TODO - ask david about what's in 'bb'
            self.bb = solps.variables['bb'].data.copy()
            self.n_cells = self.ne.size

            # process the species data into regular strings
            species_bytestrings = solps.variables['species'].data.copy()
            species = [''.join([a.decode('ASCII') for a in s]).strip() for s in species_bytestrings]
            # now separate out the ions from the neutrals
            ions = [(i, ''.join(s.split('+'))) for i, s in enumerate(species) if '+' in s]
            self.neutrals = [s for s in species if '+' not in s]
            self.ions = [i[1] for i in ions]
            # set the ion densities
            [setattr(self, f"n_{ion}", den_i[index, :, :].flatten()) for index, ion in ions]
            # set the neutral densities and temperatures
            for index, neutral in enumerate(self.neutrals):
                setattr(self, f"n_{neutral}", den_n[index, :, :].flatten())
                setattr(self, f"t_{neutral}", tab2[index, :, :].flatten())

            # find the cell centres
            crx = solps.variables['crx'].data.copy()
            cry = solps.variables['cry'].data.copy()
            self.cr = crx.mean(axis=0)
            self.cz = cry.mean(axis=0)
            R_sets = [v for v in crx.reshape([4,self.n_cells]).T]
            z_sets = [v for v in cry.reshape([4,self.n_cells]).T]

            del den_n, den_i, tab2, species_bytestrings

        R, z, triangles = connect_meshes(
            R_sets,
            z_sets,
            [triangles_from_grid([2,2])]*self.n_cells
        )

        old_Rc = zeros(2 * self.n_cells)
        old_Rc[0::2] = crx[array([0, 1, 2]), :].mean(axis=0).flatten()
        old_Rc[1::2] = crx[array([1, 2, 3]), :].mean(axis=0).flatten()

        old_zc = zeros(2 * self.n_cells)
        old_zc[0::2] = cry[array([0, 1, 2]), :].mean(axis=0).flatten()
        old_zc[1::2] = cry[array([1, 2, 3]), :].mean(axis=0).flatten()

        new_Rc = R[triangles].mean(axis=1)
        new_zc = z[triangles].mean(axis=1)

        inverse_sort = double_argsort(old_Rc, old_zc).argsort()
        descrambler = double_argsort(new_Rc, new_zc)[inverse_sort]
        triangles = triangles[descrambler,:]
        self.mesh = TriangularMesh(R=R, z=z, triangles=triangles)

    def get(self, variable, R, z, outside_value=0):
        """
        Returns the value of a chosen variable at a given set of points.

        :param variable: \
            A string indicating which variable to retrieve.

        :param R: \
            The radius values of the points at which the variable is determined.

        :param z: \
            The z-height values of the points at which the variable is determined.

        :param outside_value: \
            The value which is assigned to any points which lie outside the SOLPS grid.

        :return: \
            The values of the variable at the given points as a 1D numpy array.
        """
        inds = self.mesh.find_triangle(R, z)
        inside = inds != -1
        values = full(R.size, fill_value=outside_value)
        values[inside] = getattr(self, variable)[inds//2][inside]
        return values

    def plot(self, variable, draw_mesh=False):
        """
        Generate a colour plot of a chosen variable.

        :param str variable: A string indicating which variable to plot.
        :param bool draw_mesh: A bool indicating whether to draw the mesh in the plot.
        """
        vals = zeros(2*self.n_cells)
        vals[0::2] = getattr(self, variable)
        vals[1::2] = getattr(self, variable)
        cmap = get_cmap('viridis')

        dR = self.mesh.R_limits[1] - self.mesh.R_limits[0]
        dz = self.mesh.z_limits[1] - self.mesh.z_limits[0]
        fig = plt.figure(figsize=(8*(dR/dz)*1.3, 8))
        ax = fig.add_subplot(111)
        ax.set_facecolor(cmap(0.))
        ax.tripcolor(self.mesh.R, self.mesh.z, self.mesh.triangle_vertices, facecolors=vals)
        if draw_mesh:
            self.mesh.draw(ax, lw=0.5)
        ax.axis('equal')
        ax.set_xlabel('R (m)')
        ax.set_ylabel('z (m)')
        plt.show()


def triangles_from_grid(grid_shape):
    triangles = []
    m, n = grid_shape
    for i in range(m - 1):
        for j in range(n - 1):
            v1 = n*i + j
            v2 = n*(i+1) + j
            v3 = n*i + j + 1
            v4 = n*(i+1) + j + 1
            triangles.append([v1, v2, v3])
            triangles.append([v2, v3, v4])
    return array(triangles, dtype=int)


def double_argsort(R, z):
    z_sorter = z.argsort()
    return z_sorter[R[z_sorter].argsort()]


def connect_meshes(R_arrays, z_arrays, triangle_arrays):
    R = concatenate(R_arrays)
    z = concatenate(z_arrays)
    offsets = zeros(len(R_arrays), dtype=int)
    offsets[1:] = array([a.size for a in R_arrays], dtype=int)[:-1].cumsum()
    triangles = concatenate([tri+off for tri, off in zip(triangle_arrays, offsets)], axis=0, dtype=int)
    return remove_duplicate_vertices(R, z, triangles)
