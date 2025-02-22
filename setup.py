import setuptools
import pkg_resources

# Check pip version
try:
    import pip

    pip_version = pkg_resources.parse_version(pip.__version__)
    min_version = pkg_resources.parse_version("20.2")
    if pip_version < min_version:
        raise RuntimeError(
            f"pip version {pip_version} is too old. "
            f"Please upgrade to version 20.2 or newer and reinstall.\n"
            f"Run: `pip install --upgrade pip`"
        )
except ImportError:
    pass


setuptools.setup()
