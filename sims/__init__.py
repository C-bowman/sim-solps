from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sim-solps")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)

__all__ = ["__version__"]
