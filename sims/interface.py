
from numpy import array, diff, meshgrid, arange, concatenate, zeros
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
            # get needed info
            self.ne = solps.variables['ne'].data.flatten()
            self.te = solps.variables['te'].data.flatten() / 1.602e-19
            self.ti = solps.variables['ti'].data.flatten() / 1.602e-19

            den_i = solps.variables['na'].data
            den_n = solps.variables['dab2'].data
            self.n0 = den_n[:, :, 0:den_i.shape[-1]].flatten()
            # find the cell centres
            R_cc = solps.variables['crx'].data.mean(axis=0)
            z_cc = solps.variables['cry'].data.mean(axis=0)
            # get the indices used to define the sub-grids
            cuts = (diff(solps.variables['leftix'][0,:]) < 1).nonzero()[0]
            sep_ind = solps.variables['jsep'].data[0] + 1
            del den_n, den_i

        if cuts.size != 5:
            raise ValueError(
                """
                [ SolpsInterface error ]
                >> Currently only the 'connected double null' configuration
                >> is supported - the given balance file was of an unsupported
                >> configuration.
                """
            )

        self.grid_shape = R_cc.shape
        self.submesh_slices = {
            'lower inner leg': (slice(None, None), slice(None, cuts[0])),
            'upper inner leg': (slice(None, None), slice(cuts[1], cuts[2] + 1)),
            'upper outer leg': (slice(None, None), slice(cuts[2] + 1, cuts[3] + 1)),
            'lower outer leg': (slice(None, None), slice(cuts[4] + 1, None)),
            'lfs core': (slice(None, sep_ind + 1), slice(cuts[0], cuts[1])),
            'lfs sol': (slice(sep_ind, None), slice(cuts[0] - 1, cuts[1] + 1)),
            'hfs core': (slice(None, sep_ind + 1), slice(cuts[3] + 1, cuts[4] + 1)),
            'hfs sol': (slice(sep_ind, None), slice(cuts[3], cuts[4] + 2)),
            'upper pfr connector': tuple(meshgrid(arange(sep_ind+1), array([cuts[1], cuts[3]]))),
            'upper core connector': tuple(meshgrid(arange(sep_ind+1), array([cuts[1]-1, cuts[3]+1]))),
            'lower core connector': tuple(meshgrid(arange(sep_ind+1), array([cuts[0], cuts[4]]))),
            'lower pfr connector': tuple(meshgrid(arange(sep_ind+1), array([cuts[0]-1, cuts[4]+1]))),
            'upper x-point connector': (array([[sep_ind, sep_ind],[sep_ind, sep_ind]]), array([[cuts[1]-1, cuts[1]],[cuts[3]+1, cuts[3]]])),
            'lower x-point connector': (array([[sep_ind, sep_ind],[sep_ind, sep_ind]]), array([[cuts[0]-1, cuts[0]],[cuts[4]+1, cuts[4]]]))
        }

        R, z, triangles = connect_meshes(
            [R_cc[s].flatten() for s in self.submesh_slices.values()],
            [z_cc[s].flatten() for s in self.submesh_slices.values()],
            [triangles_from_grid(R_cc[s].shape) for s in self.submesh_slices.values()]
        )

        # to unscramble the mesh R and z, first find the mapping from the order
        # obtained from sorting by z then by R, to the original permutation
        inverse_sort = double_argsort(R_cc.flatten(), z_cc.flatten()).argsort()
        # now combine this with the sorting map for the grid R and z to get
        # a mapping straight to the original ordering
        descrambler = double_argsort(R, z)[inverse_sort]
        # now re-order the mesh data before building the mesh
        triangles = descrambler.argsort()[triangles]
        self.mesh = TriangularMesh(R_cc.flatten(), z_cc.flatten(), triangles)

    def get(self, variable, R, z):
        return self.mesh.interpolate(R, z, vertex_values=getattr(self, variable))

    def build_submeshes(self):
        R = self.mesh.R.reshape(self.grid_shape)
        z = self.mesh.z.reshape(self.grid_shape)

        return {key: TriangularMesh(
            R[slc].flatten(),
            z[slc].flatten(),
            triangles_from_grid(R[slc].shape)
            ) for key, slc in self.submesh_slices.items()
        }

    def plot(self, variable):
        r_axis, z_axis, image = self.mesh.get_field_image(getattr(self, variable), shape=(256,512))
        fig = plt.figure(figsize=(10*(r_axis.ptp()/z_axis.ptp())*1.2, 10))
        ax = fig.add_subplot(111)
        ax.set_facecolor(get_cmap('viridis')(0.))
        ax.contourf(r_axis, z_axis, image.T, 64)
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