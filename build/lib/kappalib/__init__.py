"""
This is an object-oriented plotting and inference library.
It may be imported directly, e.g.::
    import kappalib as kp

Kappalib is currently developed by Azarias, Cristiano (2020-now)
"""

import logging
# Get the version from the _version.py versioneer file. For a git checkout,
# this is computed based on the number of commits since the last tag.
from ._version import get_versions
__version__ = str(get_versions()['version'])
del get_versions

_log = logging.getLogger(__name__)