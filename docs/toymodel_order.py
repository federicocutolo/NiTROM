"""Sort key for the toymodel gallery: pipeline order, not alphabetical.

Lives in its own module (the conf dir is on ``sys.path``) so Sphinx can pickle
the environment; a function defined inside ``conf.py`` cannot be pickled.
"""

import os

_TOYMODEL_ORDER = [
    "generate_data.py",
    "train_opinf.py",
    "train_nitrom.py",
    "read_results.py",
]


def toymodel_sort_key(filename):
    """Return the pipeline position of *filename* (unknown files sort last)."""
    name = os.path.basename(filename)
    return _TOYMODEL_ORDER.index(name) if name in _TOYMODEL_ORDER else 99
