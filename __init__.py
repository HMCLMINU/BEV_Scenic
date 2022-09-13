"""A compiler and scene generator for the Scenic scenario description language.

.. raw:: html

   <h2>Submodules</h2>

.. autosummary::
   :toctree:

   core
   domains
   formats
   simulators
   syntax
"""

from .syntax.translator import scenarioFromFile, scenarioFromString

import scenic.core.errors as _errors
# import carla_birdeye_view
# from .carla_birdeye_view import *

_errors.showInternalBacktrace = False
del _errors
