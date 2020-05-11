"""Setup file for Pyrfume"""

import sys
import os

try:
    from pip.req import parse_requirements
    from pip.download import PipSession
except ImportError:
    from pip._internal.req import parse_requirements
    try:
        from pip._internal.download import PipSession
    except ImportError:
        from pip._internal.network.session import PipSession

from setuptools import setup, find_packages

# IPython 6.0+ does not support Python 2.6, 2.7, 3.0, 3.1, or 3.2
if sys.version_info < (3, 3):
    ipython = "ipython>=5.1,<6.0"
else:
    ipython = "ipython>=5.1"

def read_requirements():
    '''parses requirements from requirements.txt'''
    reqs_path = os.path.join('.', 'requirements.txt')
    install_reqs = parse_requirements(reqs_path, session=PipSession())
    reqs = [str(ir.req) for ir in install_reqs]
    return reqs

def get_version():
    version = {}
    with open("pyrfume/version.py") as f:
        exec(f.read(), version)
    return version['__version__']

setup(
    name='pyrfume',
    version=get_version(),
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
    packages=find_packages(),
    url='http://pyrfume.scidash.org',
    license='MIT',
    description='A valdiation library for human olfactory psychophysics research.',
    install_requires=read_requirements(),
    extras_require={'features': ['rdkit']}
)



