"""PICASSO."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("PICASSO")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"
