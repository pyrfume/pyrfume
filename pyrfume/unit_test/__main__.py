import sys
import unittest

from .test_docs import *
from .test_load_data import *
from .test_config_data import *
from .test_make_solutions import *
from .test_mixture import *
from .test_test import *
from .test_others_in__init__ import *
from .test_physics import *
from .test_odorants import *
from .test_datajoint_tools import DataJointTestCase

def main():
    buffer = 'buffer' in sys.argv
    database = 'database' in sys.argv
    sys.argv = sys.argv[:1]

    if not database:
        globals().pop('DataJointTestCase')

    unittest.main(buffer=buffer)

if __name__ == '__main__':
    main()
